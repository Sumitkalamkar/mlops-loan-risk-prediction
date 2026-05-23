pipeline {
    agent any

    parameters {
        choice(
            name: 'SKIP_TRAINING',
            choices: ['false', 'true'],
            description: 'Skip training and use existing model'
        )

        string(
            name: 'MIN_ACCURACY',
            defaultValue: '0.70',
            description: 'Minimum model accuracy threshold'
        )
    }

    environment {
        PROJECT_NAME = "loan-risk-prediction"
        DOCKER_IMAGE = "loan-risk-app"
        CONTAINER_NAME = "loan-risk-container"
        API_PORT = "8000"
        MLFLOW_PORT = "5000"
        AWS_DEFAULT_REGION = "ap-south-1"

        CONDA_PYTHON = "/opt/conda/envs/mlops/bin/python"
    }

    stages {

        stage('Checkout Code') {
            steps {
                echo "Cloning GitHub repository..."

                git branch: 'main',
                url: 'https://github.com/Sumitkalamkar/mlops-loan-risk-prediction.git'
            }
        }

        stage('Setup Environment') {
            steps {
                echo "Setting up Python virtual environment..."

                sh '''
                    ${CONDA_PYTHON} --version

                    ${CONDA_PYTHON} -m venv venv

                    ./venv/bin/pip install --upgrade pip setuptools wheel

                    ./venv/bin/pip install --prefer-binary -r requirements.txt --no-cache-dir

                    echo "Installed dependencies successfully"

                    ./venv/bin/python --version
                    ./venv/bin/dvc --version
                    ./venv/bin/mlflow --version
                '''
            }
        }

        stage('Configure AWS') {
            steps {
                echo "Configuring AWS region..."

                sh '''
                    mkdir -p ~/.aws

                    echo "[default]" > ~/.aws/config
                    echo "region=${AWS_DEFAULT_REGION}" >> ~/.aws/config
                '''
            }
        }

        stage('Pull Data From DVC Remote') {
            steps {
                echo "Pulling data and models from S3..."

                sh '''
                    ./venv/bin/dvc pull
                '''
            }
        }

        stage('Start MLflow Server') {
            steps {
                echo "Starting MLflow server..."

                sh '''
                    nohup ./venv/bin/mlflow ui \
                    --host 0.0.0.0 \
                    --port ${MLFLOW_PORT} \
                    > mlflow.log 2>&1 &
                '''
            }
        }

        stage('Run ML Pipeline') {

            when {
                expression {
                    params.SKIP_TRAINING == 'false'
                }
            }

            steps {
                echo "Running DVC pipeline..."

                sh '''
                    ./venv/bin/dvc repro
                '''
            }
        }

        stage('Quality Gate') {

            when {
                expression {
                    params.SKIP_TRAINING == 'false'
                }
            }

            steps {
                echo "Checking model metrics..."

                sh '''
                    ./venv/bin/python - <<EOF
import json
import sys

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics.get("accuracy", 0)

print(f"Model Accuracy: {accuracy}")

threshold = float("${MIN_ACCURACY}")

if accuracy < threshold:
    print(f"FAILED: Accuracy {accuracy} < {threshold}")
    sys.exit(1)

print(f"PASSED: Accuracy {accuracy} >= {threshold}")
EOF
                '''
            }
        }

        stage('Push Artifacts To S3') {

            when {
                expression {
                    params.SKIP_TRAINING == 'false'
                }
            }

            steps {
                echo "Pushing artifacts to S3..."

                sh '''
                    ./venv/bin/dvc push
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image..."

                sh '''
                    docker build -t ${DOCKER_IMAGE}:latest .
                '''
            }
        }

        stage('Deploy Application') {
            steps {
                echo "Deploying FastAPI container..."

                sh '''
                    docker stop ${CONTAINER_NAME} || true

                    docker rm ${CONTAINER_NAME} || true

                    docker run -d \
                    --name ${CONTAINER_NAME} \
                    -p ${API_PORT}:8000 \
                    ${DOCKER_IMAGE}:latest
                '''
            }
        }

        stage('Health Check') {
            steps {
                echo "Checking API health..."

                sh '''
                    sleep 20

                    curl http://localhost:${API_PORT}/docs
                '''
            }
        }
    }

    post {

        success {

            echo '''
===================================================
PIPELINE COMPLETED SUCCESSFULLY
===================================================

FastAPI URL:
http://YOUR_PUBLIC_IP:8000/docs

MLflow URL:
http://YOUR_PUBLIC_IP:5000

===================================================
'''
        }

        failure {

            echo '''
===================================================
PIPELINE FAILED
===================================================
'''
        }

        always {

            echo "Collecting pipeline artifacts..."

            sh '''
                mkdir -p artifacts

                cp -r metrics.json artifacts/ || true

                cp -r dvc.lock artifacts/ || true

                cp -r models/*.pkl artifacts/ || true

                cp -r mlflow.log artifacts/ || true
            '''

            archiveArtifacts artifacts: 'artifacts/**', fingerprint: true
        }
    }
}
