#Triggered by Lambda SageMaker processing job script to perform SARIMA forecasting on unemployment rate data. It reads transformed data from DynamoDB, trains and evaluates SARIMA models for each country, generates forecasts, and writes results back to DynamoDB.

import pandas as pd
import numpy as np
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


def scan_dynamodb_table(table_name):
    client = boto3.client("dynamodb", region_name="eu-central-1")
    paginator = client.get_paginator("scan")
    items = []
    for page in paginator.paginate(TableName=table_name):
        items.extend(page.get("Items", []))
    
    # Use low-level client output: items are already in DynamoDB JSON format
    def deserialize(item):
        # simple deserializer: take string or number values
        out = {}
        for k, v in item.items():
            # v is a dict like {"S": "value"} or {"N": "12.3"}
            if "S" in v:
                out[k] = v["S"]
            elif "N" in v:
                out[k] = v["N"]
            elif "BOOL" in v:
                out[k] = v["BOOL"]
            else:
                # fallback: stringify
                out[k] = list(v.values())[0]
        return out

    return pd.DataFrame([deserialize(i) for i in items])


def write_dataframe_to_dynamodb(df, table_name):
    resource = boto3.resource("dynamodb", region_name="eu-central-1")
    table = resource.Table(table_name)
    for _, row in df.iterrows():
        item = {"Country": str(row["Country"]), "TimePeriod": str(row["TimePeriod"])}
        if "Value" in row and not pd.isna(row["Value"]):
            # DynamoDB requires Decimal for numbers
            try:
                item["Value"] = Decimal(str(row["Value"]))
            except Exception:
                item["Value"] = Decimal(str(float(row["Value"])))
        if "Lower_CI" in row and not pd.isna(row["Lower_CI"]):
            item["Lower_CI"] = Decimal(str(row["Lower_CI"]))
        if "Upper_CI" in row and not pd.isna(row["Upper_CI"]):
            item["Upper_CI"] = Decimal(str(row["Upper_CI"]))
        if "RMSE" in row and not pd.isna(row["RMSE"]):
            item["RMSE"] = Decimal(str(row["RMSE"]))
        if "TrustFlag" in row and not pd.isna(row["TrustFlag"]):
            item["TrustFlag"] = str(row["TrustFlag"])
        try:
            table.put_item(Item=item)
        except Exception as e:
            print(f"⚠️ Failed to write item {item}: {e}")


def main():
    # =====================================================
    # Load Data from DynamoDB using boto3 and pandas
    # =====================================================
    dynamodb_input_table = "UnemploymentRate-ML-DynamoDB-TransformedData"
    df_sarima = scan_dynamodb_table(dynamodb_input_table)
    print(f"✅ Loaded data from DynamoDB table: {dynamodb_input_table}")

    if df_sarima.empty:
        print("No data found in input table. Exiting.")
        return

    # Keep only relevant columns if they exist
    expected_cols = ["Country", "TimePeriod", "Value", "Gender", "Category", "S_Adj"]
    for c in expected_cols:
        if c not in df_sarima.columns:
            df_sarima[c] = None

    # Filter rows
    df_sarima = df_sarima[(df_sarima["Gender"] == "T") & (df_sarima["Category"] == "TOT") & (df_sarima["S_Adj"] == "NSA")]

    df_sarima["TimePeriod"] = pd.to_datetime(df_sarima["TimePeriod"].astype(str).str.strip(), format="%Y-%m", errors="coerce")
    df_sarima["Value"] = pd.to_numeric(df_sarima["Value"], errors="coerce")
    df_sarima = df_sarima.sort_values(["Country", "TimePeriod"])

    evaluation_results = []
    forecast_results = []

    for country, group in df_sarima.groupby("Country"):
        ts = group.set_index("TimePeriod")["Value"].asfreq("MS").dropna()
        if len(ts) < 48:
            continue

        train = ts.iloc[:-12]
        test = ts.iloc[-12:]

        try:
            model = SARIMAX(
                train,
                order=(2, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted_model = model.fit(disp=False)

            test_forecast = fitted_model.get_forecast(steps=12).predicted_mean
            mae = mean_absolute_error(test, test_forecast)
            rmse = np.sqrt(mean_squared_error(test, test_forecast))

            evaluation_results.append({
                "Country": country,
                "MAE": mae,
                "RMSE": rmse,
                "Train_Size": len(train),
                "Test_Size": len(test),
            })

            final_model = SARIMAX(
                ts,
                order=(2, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            future_forecast = final_model.get_forecast(steps=6)
            future_mean = future_forecast.predicted_mean
            future_ci = future_forecast.conf_int()

            last_month = group["TimePeriod"].max()
            forecast_periods = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=6, freq="MS")
            forecast_periods_formatted = forecast_periods.strftime("%Y-%m")

            forecast_results.append(
                pd.DataFrame({
                    "Country": country,
                    "TimePeriod": forecast_periods_formatted,
                    "Value": future_mean.values,
                    "Lower_CI": future_ci.iloc[:, 0].values,
                    "Upper_CI": future_ci.iloc[:, 1].values,
                })
            )

        except Exception as e:
            print(f"⚠️ SARIMA failed for {country}: {e}")

    evaluation_df = pd.DataFrame(evaluation_results).sort_values("RMSE") if evaluation_results else pd.DataFrame()
    trust_df = evaluation_df.copy()
    if not trust_df.empty:
        trust_df["TrustFlag"] = trust_df["RMSE"] < 0.7
        trust_df = trust_df[["Country", "RMSE", "TrustFlag"]]

    if forecast_results:
        forecast_df = pd.concat(forecast_results, ignore_index=True)
    else:
        forecast_df = pd.DataFrame(columns=["Country", "TimePeriod", "Value", "Lower_CI", "Upper_CI"])

    if not forecast_df.empty and not trust_df.empty:
        forecast_df = forecast_df.merge(trust_df, on="Country", how="left")
    elif not forecast_df.empty:
        forecast_df["RMSE"] = np.nan
        forecast_df["TrustFlag"] = False

    # Prepare mixed final: last 6 months raw + forecast
    forecast_countries = forecast_df["Country"].unique() if not forecast_df.empty else []
    last_6_months = df_sarima[df_sarima["Country"].isin(forecast_countries)].copy()
    last_6_months = last_6_months.sort_values(["Country", "TimePeriod"]) if not last_6_months.empty else last_6_months
    last_6_months = last_6_months.groupby("Country", group_keys=False).tail(6).reset_index(drop=True) if not last_6_months.empty else last_6_months
    if not last_6_months.empty:
        last_6_months["TimePeriod"] = last_6_months["TimePeriod"].dt.strftime("%Y-%m")
        last_6_months["TrustFlag"] = "Raw"
        last_6_months = last_6_months[["Country", "TimePeriod", "Value", "TrustFlag"]]
    else:
        last_6_months = pd.DataFrame(columns=["Country", "TimePeriod", "Value", "TrustFlag"])

    forecast_with_flag = pd.DataFrame(columns=["Country", "TimePeriod", "Value", "TrustFlag"]) if forecast_df.empty else forecast_df[["Country", "TimePeriod", "Value", "TrustFlag"]].copy()
    if not forecast_with_flag.empty:
        forecast_with_flag["TrustFlag"] = forecast_with_flag["TrustFlag"].map({True: "True", False: "False"})

    mixed_final = pd.concat([last_6_months, forecast_with_flag], ignore_index=True) if (not last_6_months.empty or not forecast_with_flag.empty) else pd.DataFrame()
    if not mixed_final.empty:
        mixed_final = mixed_final.sort_values(["Country", "TimePeriod"]).reset_index(drop=True)

    # Write mixed_final to DynamoDB
    dynamodb_output_table = "UnemploymentRate-ML-DynamoDB-PredictedData"
    if mixed_final.empty:
        print("No forecast results to write.")
    else:
        write_dataframe_to_dynamodb(mixed_final, dynamodb_output_table)
        print(f"✅ Data successfully written to DynamoDB table: {dynamodb_output_table}")


if __name__ == "__main__":
    print("🚀 Starting SARIMA monthly job")
    main()
    print("✅ SARIMA monthly job completed")