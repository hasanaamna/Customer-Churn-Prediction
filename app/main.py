from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = "models/churn_model.pkl"
THRESHOLD_PATH = "models/churn_threshold.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CUSTOMER INPUT MODEL
# ============================================================

class CustomerData(BaseModel):

    gender: str

    SeniorCitizen: int

    Partner: str

    Dependents: str

    tenure: int

    PhoneService: str

    MultipleLines: str

    InternetService: str

    OnlineSecurity: str

    OnlineBackup: str

    DeviceProtection: str

    TechSupport: str

    StreamingTV: str

    StreamingMovies: str

    Contract: str

    PaperlessBilling: str

    PaymentMethod: str

    MonthlyCharges: float

    TotalCharges: float


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": threshold
    }


# ============================================================
# FEATURE NAME CLEANING
# ============================================================

def clean_feature_name(feature_name):

    """
    Converts technical sklearn feature names into
    cleaner names for the frontend.
    """

    name = feature_name

    # Remove pipeline prefixes
    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    # Replace encoded feature names
    name = name.replace("_", " ")

    # Convert common names
    replacements = {
        "SeniorCitizen": "Senior Citizen",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "MultipleLines": "Multiple Lines"
    }

    for old_name, new_name in replacements.items():

        name = name.replace(
            old_name,
            new_name
        )

    return name


# ============================================================
# MODEL EXPLANATION
# ============================================================

def get_model_explanation(input_df):

    try:

        # ----------------------------------------------------
        # Get preprocessing pipeline
        # ----------------------------------------------------

        preprocessor = model.named_steps["preprocessor"]


        # ----------------------------------------------------
        # Find classifier
        # ----------------------------------------------------

        classifier = None

        for step_name, step_model in model.named_steps.items():

            if hasattr(step_model, "coef_"):

                classifier = step_model

                break


        if classifier is None:

            raise ValueError(
                "Could not find a Logistic Regression "
                "classifier with coefficients."
            )


        # ----------------------------------------------------
        # Transform customer data
        # ----------------------------------------------------

        transformed_data = preprocessor.transform(
            input_df
        )


        # ----------------------------------------------------
        # Convert sparse matrix
        # ----------------------------------------------------

        if hasattr(
            transformed_data,
            "toarray"
        ):

            transformed_data = (
                transformed_data.toarray()
            )


        # First customer
        customer_values = transformed_data[0]


        # ----------------------------------------------------
        # Get feature names
        # ----------------------------------------------------

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        # ----------------------------------------------------
        # Get Logistic Regression coefficients
        # ----------------------------------------------------

        coefficients = classifier.coef_[0]


        # ----------------------------------------------------
        # Calculate contribution
        # ----------------------------------------------------

        contributions = (
            customer_values * coefficients
        )


        explanation_data = []


        for feature, contribution in zip(
            feature_names,
            contributions
        ):

            explanation_data.append(
                {
                    "feature": clean_feature_name(
                        feature
                    ),

                    "contribution": round(
                        float(contribution),
                        4
                    )
                }
            )


        # ----------------------------------------------------
        # Sort by strongest contribution
        # ----------------------------------------------------

        explanation_data.sort(
            key=lambda item:
            abs(item["contribution"]),
            reverse=True
        )


        # ----------------------------------------------------
        # Positive = increases churn probability
        # Negative = decreases churn probability
        # ----------------------------------------------------

        risk_factors = [
            item
            for item in explanation_data
            if item["contribution"] > 0
        ]


        protective_factors = [
            item
            for item in explanation_data
            if item["contribution"] < 0
        ]


        return {

            "risk_factors":
                risk_factors[:5],

            "protective_factors":
                protective_factors[:5]
        }


    except Exception as error:

        print(
            "Explanation error:",
            str(error)
        )

        return {

            "risk_factors": [],

            "protective_factors": [],

            "error":
                str(error)
        }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_churn(
    customer: CustomerData
):

    # --------------------------------------------------------
    # Convert request into dictionary
    # --------------------------------------------------------

    customer_data = customer.model_dump()


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [customer_data]
    )


    # --------------------------------------------------------
    # Generate probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_df
    )[0][1]


    # --------------------------------------------------------
    # Apply tuned threshold
    # --------------------------------------------------------

    prediction = (
        probability >= threshold
    )


    # --------------------------------------------------------
    # Convert prediction to Yes / No
    # --------------------------------------------------------

    churn_prediction = (
        "Yes"
        if prediction
        else "No"
    )


    # --------------------------------------------------------
    # Generate model explanation
    # --------------------------------------------------------

    explanation = get_model_explanation(
        input_df
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "churn_prediction":
            churn_prediction,

        "churn_probability":
            round(
                float(probability),
                4
            ),

        "threshold":
            threshold,

        "explanation":
            explanation
    }