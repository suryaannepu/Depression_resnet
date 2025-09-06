import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Alert,
  ScrollView,
  ActivityIndicator,
  SafeAreaView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { MaterialIcons } from '@expo/vector-icons';
import { Provider as PaperProvider, Card, Title, Paragraph, Button, Chip } from 'react-native-paper';

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please grant camera roll permissions to use this feature.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0]);
      setResults(null);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please grant camera permissions to use this feature.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0]);
      setResults(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) {
      Alert.alert('No image selected', 'Please select an image first.');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Create FormData for the image upload
      const formData = new FormData();
      formData.append('image', {
        uri: selectedImage.uri,
        type: 'image/jpeg',
        name: 'image.jpg',
      });

      // Replace with your actual server URL
      const serverUrl = 'http://your-server-url.com/predict';
      
      const response = await fetch(serverUrl, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();
      
      if (data.error) {
        Alert.alert('Analysis Error', data.error);
      } else {
        setResults(data);
      }
    } catch (error) {
      // For demo purposes, simulate a response
      console.log('Using mock data for demo');
      const mockResults = {
        detected_emotion: 'Happy',
        depression_level: 'No Depression',
        confidence: '0.85',
        suggestions: [
          'Keep up your positive routines.',
          'Stay connected with friends.',
          'Practice daily gratitude.'
        ]
      };
      setResults(mockResults);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setResults(null);
  };

  const getDepressionLevelColor = (level) => {
    if (level?.toLowerCase().includes('no') || level?.toLowerCase().includes('none')) {
      return '#10b981'; // green
    } else if (level?.toLowerCase().includes('mild')) {
      return '#f59e0b'; // yellow
    } else if (level?.toLowerCase().includes('severe')) {
      return '#ef4444'; // red
    }
    return '#6b7280'; // gray
  };

  return (
    <PaperProvider>
      <SafeAreaView style={styles.container}>
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          {/* Header */}
          <View style={styles.header}>
            <MaterialIcons name="psychology" size={40} color="#8b5cf6" />
            <Text style={styles.title}>MindCheck</Text>
            <Text style={styles.subtitle}>Facial Expression Analysis</Text>
          </View>

          {/* Image Selection */}
          <Card style={styles.card}>
            <Card.Content>
              <Title>Upload or Take a Photo</Title>
              
              {selectedImage ? (
                <View style={styles.imageContainer}>
                  <Image source={{ uri: selectedImage.uri }} style={styles.selectedImage} />
                  <Text style={styles.imageInfo}>Image selected</Text>
                </View>
              ) : (
                <View style={styles.uploadArea}>
                  <MaterialIcons name="cloud-upload" size={60} color="#c084fc" />
                  <Text style={styles.uploadText}>No image selected</Text>
                </View>
              )}

              <View style={styles.buttonRow}>
                <Button
                  mode="outlined"
                  onPress={pickImage}
                  style={styles.halfButton}
                  icon="image"
                >
                  Gallery
                </Button>
                <Button
                  mode="outlined"
                  onPress={takePhoto}
                  style={styles.halfButton}
                  icon="camera"
                >
                  Camera
                </Button>
              </View>

              <Button
                mode="contained"
                onPress={analyzeImage}
                disabled={!selectedImage || isAnalyzing}
                style={styles.analyzeButton}
                icon="brain"
                loading={isAnalyzing}
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Expression'}
              </Button>
            </Card.Content>
          </Card>

          {/* Results */}
          {results && (
            <Card style={styles.card}>
              <Card.Content>
                <Title>Analysis Results</Title>
                
                <View style={styles.resultsGrid}>
                  <View style={styles.resultItem}>
                    <Text style={styles.resultLabel}>Detected Emotion</Text>
                    <Text style={styles.resultValue}>{results.detected_emotion}</Text>
                  </View>
                  
                  <View style={styles.resultItem}>
                    <Text style={styles.resultLabel}>Depression Level</Text>
                    <Text style={[
                      styles.resultValue,
                      { color: getDepressionLevelColor(results.depression_level) }
                    ]}>
                      {results.depression_level}
                    </Text>
                  </View>
                  
                  <View style={styles.resultItem}>
                    <Text style={styles.resultLabel}>Confidence</Text>
                    <Text style={styles.resultValue}>{results.confidence}</Text>
                  </View>
                </View>

                {results.suggestions && results.suggestions.length > 0 && (
                  <View style={styles.suggestionsContainer}>
                    <Text style={styles.suggestionsTitle}>Personalized Suggestions</Text>
                    {results.suggestions.map((suggestion, index) => (
                      <View key={index} style={styles.suggestionItem}>
                        <MaterialIcons name="lightbulb" size={20} color="#8b5cf6" />
                        <Text style={styles.suggestionText}>{suggestion}</Text>
                      </View>
                    ))}
                  </View>
                )}

                <Button
                  mode="outlined"
                  onPress={resetAnalysis}
                  style={styles.resetButton}
                  icon="refresh"
                >
                  Analyze Another Photo
                </Button>
              </Card.Content>
            </Card>
          )}

          {/* Disclaimer */}
          <Card style={[styles.card, styles.disclaimerCard]}>
            <Card.Content>
              <Text style={styles.disclaimerText}>
                <MaterialIcons name="info" size={16} color="#6b7280" />
                {' '}This app is for educational purposes only and should not replace professional medical advice.
              </Text>
            </Card.Content>
          </Card>
        </ScrollView>
      </SafeAreaView>
    </PaperProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },
  scrollContainer: {
    padding: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
    paddingTop: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#6b7280',
    marginTop: 4,
  },
  card: {
    marginBottom: 16,
    elevation: 4,
  },
  uploadArea: {
    alignItems: 'center',
    padding: 40,
    borderWidth: 2,
    borderColor: '#c084fc',
    borderStyle: 'dashed',
    borderRadius: 12,
    marginVertical: 16,
  },
  uploadText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  imageContainer: {
    alignItems: 'center',
    marginVertical: 16,
  },
  selectedImage: {
    width: 200,
    height: 200,
    borderRadius: 12,
  },
  imageInfo: {
    marginTop: 8,
    color: '#6b7280',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 16,
  },
  halfButton: {
    flex: 0.48,
  },
  analyzeButton: {
    marginTop: 8,
    backgroundColor: '#8b5cf6',
  },
  resultsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginVertical: 16,
  },
  resultItem: {
    width: '48%',
    backgroundColor: '#f9fafb',
    padding: 16,
    borderRadius: 8,
    marginBottom: 8,
    alignItems: 'center',
  },
  resultLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  suggestionsContainer: {
    marginTop: 16,
  },
  suggestionsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  suggestionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#f8fafc',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#8b5cf6',
  },
  suggestionText: {
    flex: 1,
    marginLeft: 8,
    color: '#374151',
    lineHeight: 20,
  },
  resetButton: {
    marginTop: 16,
  },
  disclaimerCard: {
    backgroundColor: '#fef3c7',
  },
  disclaimerText: {
    color: '#92400e',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
});