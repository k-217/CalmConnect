# CalmConnect: Your Mental Health Companion

Calm Connect is a fundamental Retrieval-Augmented Generation based app that is designed to provide mental consultation or simply someplace in the artificial universe where one can open up and share valid experiences. It provides real-time answers to all the user asks and also does not judge you for every emotion. It suggests pratical techniques for managing stress and anxiety. Have faith and it will be your perfect friend in the times of need! Happy Chatting!

## Features

- **Privacy-First Approach**: 
  - Utilizes PII detection to identify and avoid processing sensitive user information.
  - Ensures conversations remain confidential and secure.

- **Support and Companionship**: 
  - Engages users in supportive dialogues that promote mental well-being.
  - Offers resources, coping strategies, and encouragement based on user input.

- **Non-Medical Advice**: 
  - Provides general guidance and support without offering medical diagnoses or treatment.
  - Directs users to appropriate resources or professionals if needed.

- **Feedback System**:
  - Allows users to provide feedback with an integer rating scale for the responses provided by the app.

## Files Included

- `data/`: Contains datasets and resources for the application. Organized for easy access and modification.
- `index.py`: The main application code to run the bot and handles user interactions. Contains the core logic for processing inputs and generating responses
- `requirements.txt`: Lists the dependencies required for the project.
- `app.yaml`: Configuration file for deployment, detailing environment settings and runtime configurations. Essential for hosting services like Google Drive.
- `.env`: Environment variables for the application (example: API key, GPT Model)
- `DockerFile`: Instructions to build a Docker image for the application. Facilitates containerization, making deployment easier across various environments.

## Demo Video

The video for demo may be found [here](). 

## Future Projections

- **Database Integration**: Store user sessions and feedback in a database for persistence.
- **Advanced PII Detection**: Implement more robust models or services for PII detection.
- **User Authentication**: Ensure user sessions are secure with authentication measures.
- **Natural Language Understanding**: Improve the understanding of user intents and emotions.

## Installation

To set up the CalmConnect locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/k-217/CalmConnect
   cd CalmConnect
2. **Install Required Packages:**:
   ```bash
   pip install -r requirements.txt
   cd CalmConnect
3. **Run the Application**:
   ```bash
   index.py
   cd CalmConnect

## Believe in Yourself! CalmConnect stands behind you! Stay Strong!
