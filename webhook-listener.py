from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

# --- IMPORTANT ---
# Change 'YOUR_REPO_NAME' to the name of your app's folder
# (e.g., 'my-gitops-app')
# -----------------
REPO_FOLDER_NAME = 'USP_MiniProject'

# Get the full path to the deploy script
DEPLOY_SCRIPT_PATH = os.path.join(
    os.getcwd(), 
    REPO_FOLDER_NAME, 
    'deploy.sh'
)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    print("Webhook received!")

    # Run the deployment script
    try:
        # Run the script and capture output
        result = subprocess.run(
            [DEPLOY_SCRIPT_PATH], 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=os.path.join(os.getcwd(), REPO_FOLDER_NAME) # Run script from repo dir
        )
        print(result.stdout)
        print("Deployment script executed successfully.")
        return jsonify({'status': 'success'}), 200
    except subprocess.CalledProcessError as e:
        # If the script fails, log the error
        print(f"Deployment failed:\n{e.stderr}")
        return jsonify({'status': 'error', 'message': e.stderr}), 500
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Runs on port 5001 so it doesn't conflict with your app on 8080
    app.run(host='0.0.0.0', port=5001)
