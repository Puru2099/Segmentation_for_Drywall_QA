Prompted Segmentation for Drywall Quality AssuranceThis repository contains the code and models for a project focused on the automated quality assurance of drywall. The primary goal is to segment two key features from images: cracks and taping areas. The project includes data preparation scripts, multiple model training experiments, and a fully interactive web application to demonstrate the final model.Table of ContentsProject OverviewFeaturesFinal Model PerformanceDirectory StructureSetup and InstallationUsageModel Development and ExperimentsProject OverviewThe core of this project is a machine learning pipeline that takes an image of drywall as input and produces a binary mask highlighting defects (cracks) or features (taping). This is achieved by fine-tuning a state-of-the-art segmentation model on a custom, augmented dataset.The final deliverable is an interactive web application that allows users to upload their own images, provide point prompts, and receive a segmentation mask generated in real-time by the fine-tuned Segment Anything Model 2.1 (SAM 2.1).FeaturesInteractive Web UI: A user-friendly interface built with React and TypeScript for easy image upload and interaction.High-Performance Backend: A Flask (Python) backend serves the fine-tuned SAM 2.1 model for fast inference.Point-Prompted Segmentation: Leverages the power of SAM 2.1 to generate precise masks from simple point grids.Comprehensive Data Pipeline: Includes scripts for converting COCO annotations and performing extensive offline data augmentation.Multi-Model Experimentation: The project explored five different model architectures, with the best-performing one selected for the final application.Final Model PerformanceThe final model deployed in the web application is a fine-tuned SAM 2.1 (Hiera Tiny) model. It was selected after a comparative analysis of five different approaches.MetricTest Set ScoreMean IoU (mIoU)0.6407Dice Score0.7747Directory Structure/
├── backend/
│   ├── app.py              # Flask server application
│   ├── model.py            # Model loading and prediction logic
│   ├── requirements.txt    # Python dependencies
│   ├── sam2.1_hiera_tiny.pt # Base SAM 2.1 model weights
│   └── sam2.1_best_e6_miou0.6201.pt # Fine-tuned model weights
│
├── components/             # React components for the UI
│   ├── ImageUploader.tsx
│   ├── PromptControls.tsx
│   └── ResultDisplay.tsx
│
├── services/               # Frontend service for API calls
│   └── geminiService.ts
│
├── App.tsx                 # Main React application component
├── index.html              # Entry point for the web app
├── package.json            # Node.js project configuration
└── README.md               # This file
Setup and InstallationFollow these steps to set up and run the project locally.PrerequisitesPython 3.9+Node.js and npm (or yarn)PyTorch and Torchvision (see backend/requirements.txt)1. Backend SetupFirst, set up the Python backend server.# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt

# Download the model weights if they are not present
# (Ensure sam2.1_hiera_tiny.pt and sam2.1_best_e6_miou0.6201.pt are in this directory)
2. Frontend SetupNext, set up the React frontend.# From the project root directory
# Install Node.js dependencies
npm install

# (Optional) Create a .env file in the root for any environment variables
Usage1. Start the Backend ServerWith your virtual environment activated, run the Flask application from the backend directory.cd backend
python app.py
The backend server will start on http://localhost:5000.2. Start the Frontend ApplicationIn a new terminal, run the React development server from the project's root directory.npm run dev
The web application will be available at http://localhost:3000. Open this URL in your browser to use the application.How to Use the AppUpload an Image: Click the upload area or drag and drop a drywall image.Provide a Prompt: The application uses an automatic grid of points as prompts, so you can simply click the "Generate Mask" button.View Results: The original image and the generated segmentation mask will be displayed side-by-side.Model Development and ExperimentsThis project involved a thorough investigation into the best model for the task. The dataset was created by combining two Roboflow datasets (drywall-join-detect and cracks-3ii36) and expanding it to 18.5k images via offline augmentation.Five distinct models were trained and evaluated:ApproachModelPromptingFinal mIoUFinal Dice1CLIPSegText0.56250.71062SAM 2.1 (Final)Point Grid0.64070.77473SAM 2.1 (Improved Loss)Point Grid0.70240.80164SegFormer B2None (Semantic)0.65910.77065YOLOE-LNone (Semantic)0.53510.6683While Approach 3 showed the highest validation scores, the model weights were not recoverable. Therefore, the robust and high-performing model from Approach 2 was selected for the final application.
