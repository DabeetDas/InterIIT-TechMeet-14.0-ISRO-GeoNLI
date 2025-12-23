# Deployment script for the application

# Step 1: Deploy modal GPU server for the backend
cd backend
pip install modal
modal setup # this will redirect to the modal website to login
modal deploy script.py # this will deploy modal and give a URL for accesing deployed models on A100 GPUs

# Update the modal URL in the backend server
sed -i 's|https://justharshitjaiswal14--vlm-inference-server-fastapi-e-17eb43-dev.modal.run|https://your-actual-modal-url.modal.run|g' modal_request.py

# Step 2: Deploy the backend server
pip install -r requirements.txt
python main.py
cd ..

# Step 2: Deploy the frontend
cd TRINETRA
npm install
npm run dev
cd ..