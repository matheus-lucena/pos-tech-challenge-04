import os
import json
from typing import Dict, Any, Optional

import boto3


class SageMakerService:
    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        self.endpoint_name = endpoint_name or os.getenv("SAGEMAKER_ENDPOINT")
        if not self.endpoint_name:
            raise ValueError(
                "SAGEMAKER_ENDPOINT not configured. "
                "Set the SAGEMAKER_ENDPOINT environment variable or pass endpoint_name to the constructor."
            )
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("sagemaker-runtime", region_name=self.region_name)

    def predict_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(data),
            )
            result = json.loads(response["Body"].read().decode())
            status = "HIGH RISK" if result.get("maternal_health_risk") == 1 else "LOW RISK"
            return {
                "status": status,
                "risk_probability": result.get("risk_probability", 0),
                "raw_result": result,
            }
        except Exception as e:
            raise Exception(f"SageMaker prediction error: {str(e)}")
