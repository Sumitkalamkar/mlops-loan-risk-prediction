pipeline {
    agent { label "ec2" }

    parameters {
        choice(
            name: 'SKIP_TRAINING',
            choices: ['false', 'true'],
            description: 'Skip training and use existing model (for deployment only)'
        )
        string(
            name: 'MIN_ACCURACY',
            defaultValue: '0.70',
            description: 'Minimum model accuracy threshold for quality gate'
        )
    }

    environment {
        AWS_DEFAULT_REGION = 'ap-south-1'
        MLFLOW_TRACKING_URI = 'http://localhost:5000'
        MIN_MODEL_ACCURACY = "${params.MIN_ACCURACY}"
    }

    stages {
        stage('Setup Environment') {
            steps {
                echo "Setting up Python environment..."
                sh '''
                    # Install system dependencies
                    yum install -y python3 python3-pip docker git
                    
                    # Create virtual environment
                    python3 -m venv venv
                    
                    # Upgrade pip and install dependencies
                    ./venv/bin/pip install --upgrade pip setuptools wheel
                    ./venv/bin/pip install -r requirements.txt --no-cache-dir
                    ./venv/bin/pip install setuptools==69.5.1 wheel==0.42.0 packaging==24.0
                    
                    # Verify installations
                    echo "DVC Version: $(./venv/bin/dvc --version)"
                    echo "MLflow Version: $(./venv/bin/mlflow --version)"
                '''
            }
        }

        stage('Pull Data from DVC Remote') {
            steps {
                echo "Pulling raw data from S3 via DVC..."
                sh '''
                    # Pull only raw data (DVC tracked files)
                    ./venv/bin/dvc pull data/loan_data.csv.dvc
                    
                    # Verify data was pulled
                    if [ -f "data/loan_data.csv" ]; then
                        echo "Raw data pulled successfully"
                        echo "Rows: $(wc -l < data/loan_data.csv)"
                    else
                        echo "Failed to pull data from DVC remote"
                        exit 1
                    fi
                '''
            }
        }

        stage("Start Mlflow server") {
            steps {
                echo "Starting MLflow server in background..."
                sh '''
                    # Start MLflow server in background
                    nohup ./venv/bin/mlflow server \
                    --backend-store-uri sqlite:///mlflow.db \
                    --default-artifact-root ./mlruns \
                    --host 0.0.0.0 \
                    --port 5000 & 
                    ''' 
                }
        }


        stage('Run ML Pipeline') {
            when {
                expression { params.SKIP_TRAINING != 'true' }
            }
            steps {
                echo "Running DVC reproducible pipeline..."
                sh '''
                    # Create necessary directories
                    mkdir -p models data/processed
                    
                    # Run the full DVC pipeline
                    # This executes: validation -> cleaning -> features -> train -> evaluate
                    # DVC caches outputs - unchanged stages won't re-run
                    ./venv/bin/dvc repro
                    
                    # Show pipeline execution graph
                    echo ""
                    echo "Pipeline DAG:"
                    ./venv/bin/dvc dag
                    
                    # Show what changed
                    echo ""
                    echo "Pipeline Status:"
                    ./venv/bin/dvc status
                '''
            }
        }

        stage('Quality Gate') {
            steps {
                echo "Checking model quality metrics..."
                sh '''
                    if [ -f "metrics.json" ]; then
                        echo "Model Metrics:"
                        cat metrics.json | python3 -m json.tool
                        
                        # Extract accuracy and check threshold
                        ACCURACY=$(python3 -c "import json; print(json.load(open('metrics.json'))['accuracy'])")
                        THRESHOLD=${MIN_MODEL_ACCURACY}
                        
                        echo ""
                        echo "Accuracy: $ACCURACY (threshold: $THRESHOLD)"
                        
                        PASS=$(python3 -c "print('yes' if $ACCURACY >= $THRESHOLD else 'no')")
                        if [ "$PASS" = "no" ]; then
                            echo "Model failed quality gate!"
                            exit 1
                        fi
                        echo "Model passed quality gate!"
                    else
                        echo "Warning: No metrics.json found - skipping quality check"
                    fi
                '''
            }
        }

        stage('Push Artifacts to S3') {
            steps {
                echo "Pushing all artifacts to S3 via DVC..."
                sh '''
                    # Push all DVC tracked outputs to S3
                    # This includes: processed data, models, scalers
                    ./venv/bin/dvc push
                    
                    if [ $? -eq 0 ]; then
                        echo "DVC push completed successfully"
                        
                        # Show what's in the remote
                        echo ""
                        echo "Cloud Status:"
                        ./venv/bin/dvc status --cloud
                    else
                        echo "DVC push failed"
                        exit 1
                    fi
                '''
            }
        }

        stage('Commit Pipeline State to Git') {
            steps {
                echo "Committing dvc.lock back to repository..."
                    sh '''
                        # Configure git
                        git config user.email "jenkins@ci.local"
                        git config user.name "Jenkins CI"
                        git remote add origin-ssh  git@github.com:sudhanshuvlog/mlops-project.git || true
                        ssh-keyscan github.com >> ~/.ssh/known_hosts || true
                        
                        # Check if there are changes to commit
                        if git diff --quiet dvc.lock 2>/dev/null && git diff --quiet metrics.json 2>/dev/null; then
                            echo "No changes to dvc.lock or metrics.json - skipping commit"
                            # Still create tag for this build
                            TAG_NAME="model-v${BUILD_NUMBER}"
                        else
                            # Add pipeline state files
                            git add dvc.lock metrics.json
                            
                            # Commit with build info
                            git commit -m "chore: Update pipeline state [Jenkins Build #${BUILD_NUMBER}]

                        - Updated dvc.lock with latest pipeline run
                        - Updated metrics.json with model performance
                        - Triggered by: ${BUILD_URL}

                        [skip ci]" || echo "Nothing to commit"
                                                    
                                                # Push commit
                                                git push origin-ssh HEAD:master
                                                fi
                                                
                                                # Create and push a tag for this model version
                                                # This tag points to the CORRECT commit with dvc.lock
                                                TAG_NAME="model-v${BUILD_NUMBER}"
                                                git tag -a ${TAG_NAME} -m "Model version from Jenkins Build #${BUILD_NUMBER}

                        Metrics: $(cat metrics.json 2>/dev/null || echo 'N/A')
                        To reproduce: git checkout ${TAG_NAME} && dvc pull"
                        
                        git push origin-ssh ${TAG_NAME}
                        
                        echo "======================================="
                        echo "Tagged as: ${TAG_NAME}"
                        echo "To reproduce this model:"
                        echo "  git checkout ${TAG_NAME}"
                        echo "  dvc pull"
                        echo "======================================="
                    '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image..."
                sh '''
                    # Build with cache
                    docker build -t loan-risk-app:${BUILD_NUMBER} -t loan-risk-app:latest .
                    
                    echo "Docker image built: loan-risk-app:${BUILD_NUMBER}"
                '''
            }
        }

        stage('Deploy Application') {
            steps {
                echo "Deploying application container..."
                sh '''
                    # Stop existing container
                    docker rm -f webapp || true
                    
                    # Run new container
                    docker run -d \
                        --name webapp \
                        -p 8000:8000 \
                        -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
                        loan-risk-app:${BUILD_NUMBER}
                    
                    # Wait for container to start
                    sleep 5
                    
                    # Health check
                    echo "Running health check..."
                    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ || echo "000")
                    
                    if [ "$HEALTH" = "200" ]; then
                        echo "Application deployed and healthy!"
                    else
                        echo "Warning: Health check returned: $HEALTH"
                        docker logs webapp
                    fi
                '''
            }
        }
    }

    post {
        always {
            echo "Collecting pipeline artifacts..."
            sh '''
                mkdir -p artifacts
                cp -r metrics.json dvc.lock artifacts/ 2>/dev/null || true
                cp -r models/*.pkl artifacts/ 2>/dev/null || true
            '''
            archiveArtifacts artifacts: 'artifacts/**', allowEmptyArchive: true
        }
        
        success {
            echo '''
            ===================================================
            PIPELINE COMPLETED SUCCESSFULLY!
            ===================================================
            
            Metrics: Check artifacts/metrics.json
            API: http://<server-ip>:8000
            Artifacts pushed to S3 via DVC
            
            To pull artifacts on another machine:
              dvc pull
            
            ===================================================
            '''
        }
        
        failure {
            echo '''
            ===================================================
            PIPELINE FAILED!
            ===================================================
            '''
        }
    }
}