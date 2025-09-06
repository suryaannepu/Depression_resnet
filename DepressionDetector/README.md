# MindCheck - Depression Detection App

A React Native mobile application that analyzes facial expressions to detect potential signs of depression using a ResNet deep learning model.

## Features

- 📸 Take photos or select from gallery
- 🧠 AI-powered facial expression analysis
- 📊 Depression level assessment (None, Mild, Severe)
- 💡 Personalized suggestions based on results
- 📱 Clean, intuitive mobile interface

## Setup Instructions

### Prerequisites
- Node.js (v14 or higher)
- Expo CLI (`npm install -g @expo/cli`)
- Android Studio (for APK building)

### Installation

1. Install dependencies:
```bash
cd DepressionDetector
npm install
```

2. Start the development server:
```bash
npx expo start
```

3. To build APK:
```bash
# Install EAS CLI
npm install -g @expo/cli

# Configure EAS
npx eas build:configure

# Build APK
npx eas build --platform android --profile preview
```

### Server Configuration

The app expects a backend server running the Flask application. Update the `serverUrl` in `App.js`:

```javascript
const serverUrl = 'http://your-server-url.com/predict';
```

For local development, you can use ngrok to expose your local Flask server:
```bash
# Install ngrok
npm install -g ngrok

# Expose local server (assuming Flask runs on port 5000)
ngrok http 5000
```

## Model Information

This app uses a ResNet-50 model trained to classify facial expressions into 7 categories:
- Angry
- Disgust  
- Fear
- Happy
- Neutral
- Sad
- Surprise

The model then maps these emotions to depression levels:
- **No Depression**: Happy, Surprise, Neutral
- **Mild Depression**: Fear, Disgust
- **Severe Depression**: Sad, Angry

## Important Notes

⚠️ **Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for mental health concerns.

## Building APK

To generate a downloadable APK file:

1. Create an Expo account at https://expo.dev
2. Install EAS CLI: `npm install -g @expo/cli`
3. Login: `npx eas login`
4. Configure build: `npx eas build:configure`
5. Build APK: `npx eas build --platform android --profile preview`

The build process will provide a download link for your APK file.

## License

This project is for educational purposes. Please ensure you have proper permissions for any models or datasets used.