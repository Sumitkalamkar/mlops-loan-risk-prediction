# ML Model Deployment: Loan Risk Prediction

A complete MLOps pipeline for training, versioning, and deploying a machine learning model (XGBoost) with **DVC** for artifact versioning, **MLflow** for experiment tracking, **Jenkins** for CI/CD automation & Docker for running the prediction webapp.

## Architecture Overview

```
┌─────────────────────┐
│  Git Repository     │ (dvc.yaml, dvc.lock, code)
└──────────┬──────────┘
           │
      ┌────▼────┐
      │ Jenkins  │ (CI/CD Pipeline)
      └────┬────┘
           │
    ┌──────┴──────┬─────────────────┐
    │             │                 │
    ▼             ▼                 ▼
┌────────┐  ┌──────────┐  ┌─────────────────┐
│ Train  │→ │ DVC Push │→ │ AWS S3 Storage  │
│ Model  │  │ Artifacts│  │ (models, data)  │
└────────┘  └──────────┘  └────────┬────────┘
    │                              │
    ▼                              │
┌────────────┐                     │
│ MLflow     │                     │
│ Dashboard  │                     │
└────────────┘                     │
                                   │
                        ┌──────────▼────────────┐
                        │ Docker Container      │
                        │ (FastAPI + predict.py)│
                        │ - dvc pull from S3    │
                        │ - Load models         │
                        │ - Serve API           │
                        └──────────────────────┘
```

## Model Training Steps

```

         +------------------------+              
         | data/loan_data.csv.dvc |              
         +------------------------+              
               ***          **                   
              *               **                 
            **                  **               
+-----------------+         +------------+       
| data_validation |         | clean_data |       
+-----------------+         +------------+       
                                   *             
                                   *             
                                   *             
                        +---------------------+  
                        | feature_engineering |  
                        +---------------------+  
                                    *              
                                    *                 
                                    *                  
                                +-------+                 
                                | train |                  
                                +-------+                  
                                    *              
                                    *              
                                    *           
                                +----------+        
                                | evaluate |        
                                +----------+   
```

![Pipeline](pipeline.png)
---

## Prerequisites

### Local Machine
- Python 3.9+
- Git
- AWS account with S3 access
- Docker

### AWS Setup
- **S3 Bucket**: `mlops-loan-risk-gfg`
- **IAM Role** to provide ec2 jenkins worker node access to aws s3
- **EC2 Instance** for Jenkins agent (t2.medium recommended)

### Jenkins Server
- Docker or standalone Jenkins
- JDK 21
- Docker installed on Jenkins agent

---

## Part 1: Local Development Setup

### Step 1: Clone and Create Virtual Environment  (this needs to be done in our local system)

```bash
git clone <repo-url>
cd mlops-project

# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies  (this needs to be done in our local system)

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 3: Configure AWS Credentials On Local System For Testing  (this needs to be done in our local system)

```bash
# Set environment variables or use AWS CLI
aws configure

# Or set directly:
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=ap-south-1
```

### Step 4: Initialize DVC and Set Remote Storage  (this needs to be done in our local system)

```bash
# Initialize DVC (one-time)
dvc init --no-scm

# Add S3 as remote storage
dvc remote add -d myremote s3://mlops-loan-risk-gfg/dvc-storage
```

### Step 5: Generate Training Data  (this needs to be done in our local system)

```bash
python data-generator.py
# Creates: data/loan_data.csv
```

### Step 6: Run dvc repro  (this needs to be done in our local system)

```bash
dvc repro
```

### Step 7: Push Artifacts to S3 (this needs to be done in our local system)

```bash
dvc push

# Verify
dvc status --cloud
```

## Part 2: Jenkins CI/CD Pipeline Setup

### Prerequisites
- Jenkins server running (Docker or standalone) - `docker run -p 8080:8080 -p 50000:50000 -dit --name jenkins --restart=on-failure -v jenkins_home:/var/jenkins_home jenkins/jenkins:lts-jdk21`
- EC2 agent with Python 3.9+, Docker, Git - Run the below Command to download java JDK

`wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.rpm`
`yum install jdk-21_linux-x64_bin.rpm -y`

### Step 1: Create IAM Role to Provide EC2 Instance access to s3
- Create IAM Role -> Go to AWS → IAM → Roles → Create Role:
- Trusted entity: EC2
- Attach policy like:

```json
{
    "Version": "2026-05-09",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::mlops-loan-risk-gfg"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::mlops-loan-risk-gfg/*"
        }
    ]
}
```

- Attach Role to EC2 (Jenkins Worker)
- EC2 → Instance → Actions → Security → Modify IAM Role - Attach the role

### Step 2: Setup Github SSH Keys For Pushing The Dvc.lock From The Jenkins Pipeline

- `ssh-keygen -t rsa` - to create key pairs from jenkins ec2 worker node
- Go to github `repo deployed keys` add the public key over there, by providing the push access (this is required)

### Step 3: Create Jenkins Pipeline Job

1. Jenkins → **New Item** → **Pipeline**
2. Name: `loan-risk-pipeline`
3. Pipeline → Definition: **Pipeline script from SCM**
   - SCM: Git
   - Repository URL: `<your-repo-url>`
   - Branch: `*/main`
   - Script Path: `Jenkinsfile`
4. **Save**

### Step 4: Configure Build Triggers (Optional)

For auto-trigger on Git push:
- GitHub: Configure webhook to send to Jenkins
- Or: Poll SCM (`H/5 * * * *` every 5 min)

### Step 5: Run Pipeline

Click **Build** or **Build with Parameters** (if defined):
- `SKIP_TRAINING`: false (train) or true (reuse S3)

### What Happens in Pipeline

```
1. Setup Environment    → Create venv, install deps
2. DVC Setup           → Initialize DVC, fetch from S3 (if dvc.lock exists)
3. Start MLflow Server → MLflow UI on port 5000
4. Train with DVC      → dvc repro (preprocess + train)
5. Collect Metrics     → Display dvc status, MLflow runs
6. Push to S3          → dvc push (upload artifacts)
7. Build Docker Image  → Create Docker image
8. Run Container       → Start API on port 8000
9. Health Check        → Verify API is responding
```

---

## Part 3: DVC & MLflow Workflow

### DVC: Data Versioning & Pipelines

**Track artifacts:**
```bash
# View pipeline
dvc dag

# Check status
dvc status

# Push to S3
dvc push

# Pull from S3 (on another machine)
dvc pull

# View history
git log --oneline  # shows dvc.lock changes
```

**Rollback to previous version:**
```bash
# 1. Find commit with desired dvc.lock
git log --oneline -- dvc.lock

# 2. Checkout that version
git checkout <commit-hash> -- dvc.lock

# 3. Sync DVC workspace
dvc checkout

# 4. Rebuild/redeploy
```

### MLflow: Experiment Tracking

**View experiments:**
```bash
# Local UI
nohup ./venv/bin/mlflow server \
 --backend-store-uri sqlite:///mlflow.db \
 --default-artifact-root ./mlruns \
 --host 0.0.0.0 \
 --port 5000 &
# Open: http://ec2-ip:5000
```

**Compare runs:**
- MLflow UI shows all training runs
- Compare metrics: accuracy, precision, recall, F1
- View hyperparameters used
- Track S3 paths of artifacts

**Register model:**
- In MLflow UI: Runs → Click run → Register Model
- Creates versioned model entry (for production serving)

---

## Part 4: Docker Deployment

### Local Testing

```bash
# Build image
docker build -t loan-risk-app .

# Run container
docker run -d \
  -e AWS_DEFAULT_REGION=ap-south-1 \
  -p 8000:8000 \
  --name webapp \
  loan-risk-app

# Test API
curl http://localhost:8000/
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 50000, "loan_amount": 20000, "credit_score": 700}'

# View logs
docker logs -f webapp

# Stop
docker stop webapp
docker rm webapp
```

### Docker Image Details

- **Base**: Python 3.10-slim
- **Entrypoint**: `dvc pull` + start API
- **Port**: 8000

---

## Part 5: API Usage

### Prediction Endpoint

**Request:**
```bash
POST /predict
Content-Type: application/json

{
  "income": 50000,
  "loan_amount": 20000,
  "credit_score": 700
}
```

**Response:**
```json
{
  "risk": "low",
  "confidence": 0.95
}
```

### Health Check

```bash
GET /
# Response: {"message": "Loan Risk API Running"}
```

---

## Part 6: Model Training & Evaluation

### Training Pipeline (`dvc repro`)

1. **Preprocess** → loads `data/loan_data.csv`
   - Drop NaNs
   - Encode target variable (risk)
   - Scale features (StandardScaler)
   - Outputs: `models/scaler.pkl`, `models/label_encoder.pkl`

2. **Train** → trains XGBClassifier
   - Test/Train split: 80/20
   - Logs metrics to MLflow
   - Outputs: `models/model.pkl`
   - Min accuracy threshold: 0.8 (fails if below)

### Metrics Tracked

- **Accuracy**: overall correctness
- **Precision**: true positives / predicted positives
- **Recall**: true positives / actual positives
- **F1 Score**: harmonic mean of precision & recall


## Part 7: Model Versioning & Rollback

### Version Tracking

- **Git**: tracks `dvc.lock` (reproducible state)
- **S3**: stores actual model files (via DVC)
- **MLflow**: logs all experiment runs

### Rollback to Previous Version

**Option 1: Via DVC Lock**
```bash
# View history
git log --oneline -- dvc.lock

# Checkout previous version
git checkout <commit-hash> -- dvc.lock

# Sync workspace
dvc checkout

# Redeploy
```

---

## Part 8: Common Commands

### Development
```bash
# Run training locally
dvc repro

# Start MLflow dashboard
mlflow ui --host 0.0.0.0 --port 5000

# Push to S3
dvc push

# Pull from S3
dvc pull

# View pipeline
dvc dag

# Test prediction
python -c "from src.predict import predict; print(predict([50000, 20000, 700]))"
```

### Docker
```bash
# Build image
docker build -t loan-risk-app .

# Run container
docker run -d -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -p 8000:8000 loan-risk-app

# Test API
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"income": 50000, "loan_amount": 20000, "credit_score": 700}'

# View logs
docker logs -f webapp
```

### Jenkins
```bash
# View pipeline execution
Jenkins → <pipeline> → Build History

# Trigger new build
Jenkins → <pipeline> → Build Now (or Build with Parameters)

# View logs
Jenkins → Build → Console Output
```

---

## Part 9: Troubleshooting

### DVC Issues

**Error: `cannot import name '_DIR_MARK'`**
```bash
pip install --upgrade dvc[s3] pathspec
```

**Error: `dvc pull` fails**
```bash
# Verify AWS credentials
aws s3 ls s3://mlops-loan-risk-gfg/

# Check DVC remote
dvc remote list

# Reconfigure if needed
dvc remote remove myremote
dvc remote add -d myremote s3://mlops-loan-risk-gfg/dvc-storage
```

---

## Environment Variables

### AWS Credentials
```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=ap-south-1
```

### MLflow
```
MLFLOW_TRACKING_URI=http://localhost:5000  # or remote server
```

### DVC
```
DVC_REMOTE=myremote
```

---

## Key Files

| File | Purpose |
|------|---------|
| `dvc.yaml` | Pipeline definition (preprocess, train stages) |
| `dvc.lock` | Reproducible state (artifact hashes) |
| `Jenkinsfile` | CI/CD pipeline steps |
| `Dockerfile` | Container configuration |
| `requirements.txt` | Python dependencies |
| `src/train.py` | Training script (MLflow logging) |
| `src/predict.py` | Prediction logic |
| `src/preprocess.py` | Data preprocessing |
| `src/app.py` | FastAPI application |

---

## References

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Docker Documentation](https://docs.docker.com/)

---

## Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Review Jenkins logs: `Jenkins → Build → Console Output`
3. Check container logs: `docker logs webapp`
4. Verify AWS credentials: `aws s3 ls s3://mlops-loan-risk-gfg/`
