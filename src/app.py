from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()


class LoanRequest(BaseModel):
    income: float
    loan_amount: float
    credit_score: float


@app.get("/")
def home():
    return {"message": "Loan Risk API Running"}


@app.post("/predict")
def get_prediction(req: LoanRequest):
    data = [req.income, req.loan_amount, req.credit_score]
    pred, prob = predict(data)

    return {
        "risk": str(pred),
        "confidence": prob
    }