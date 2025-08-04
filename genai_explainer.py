import openai
import os
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnExplanationGenerator:
    """
    Generative AI module for explaining churn predictions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the explanation generator
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        else:
            logger.warning("OpenAI API key not provided. Explanations will be limited.")
        
        self.model_name = "gpt-3.5-turbo"
        self.max_tokens = 500
        
    def generate_explanation(self, 
                           prediction: int, 
                           probability: float,
                           customer_data: Dict,
                           feature_importance: Dict,
                           model_type: str = "Random Forest") -> Dict:
        """
        Generate a natural language explanation for a churn prediction
        
        Args:
            prediction: Predicted churn (0 or 1)
            probability: Prediction probability
            customer_data: Customer features and values
            feature_importance: Feature importance scores
            model_type: Type of ML model used
            
        Returns:
            Dictionary containing the explanation
        """
        try:
            if not self.api_key:
                return self._generate_simple_explanation(prediction, probability, customer_data, feature_importance)
            
            # Prepare the prompt
            prompt = self._create_explanation_prompt(prediction, probability, customer_data, feature_importance, model_type)
            
            # Generate explanation using OpenAI
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist specializing in customer churn analysis. Provide clear, business-friendly explanations of churn predictions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            
            return {
                "prediction": prediction,
                "probability": probability,
                "explanation": explanation,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
                "generation_method": "OpenAI GPT"
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return self._generate_simple_explanation(prediction, probability, customer_data, feature_importance)
    
    def _create_explanation_prompt(self, 
                                 prediction: int, 
                                 probability: float,
                                 customer_data: Dict,
                                 feature_importance: Dict,
                                 model_type: str) -> str:
        """
        Create a detailed prompt for the AI model
        """
        churn_status = "likely to churn" if prediction == 1 else "likely to stay"
        confidence = "high" if probability > 0.8 else "medium" if probability > 0.6 else "low"
        
        # Get top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        top_features = [f"{feature}: {importance:.3f}" for feature, importance in sorted_features]
        
        # Format customer data for explanation
        customer_info = []
        for feature, value in customer_data.items():
            if feature in feature_importance:
                customer_info.append(f"{feature}: {value}")
        
        prompt = f"""
        Based on the following information, provide a clear and business-friendly explanation for a customer churn prediction:

        PREDICTION: Customer is {churn_status} (confidence: {confidence}, probability: {probability:.2%})
        MODEL: {model_type}
        
        CUSTOMER DATA:
        {chr(10).join(customer_info)}
        
        TOP FEATURES (by importance):
        {chr(10).join(top_features)}
        
        Please provide an explanation that:
        1. Explains the prediction in simple business terms
        2. Highlights the most important factors contributing to this prediction
        3. Suggests potential actions to improve customer retention (if churn is predicted)
        4. Uses a professional but accessible tone
        5. Is 2-3 sentences long
        
        Explanation:
        """
        
        return prompt
    
    def _generate_simple_explanation(self, 
                                   prediction: int, 
                                   probability: float,
                                   customer_data: Dict,
                                   feature_importance: Dict) -> Dict:
        """
        Generate a simple explanation without OpenAI API
        """
        churn_status = "likely to churn" if prediction == 1 else "likely to stay"
        
        # Get top 3 features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation_parts = []
        explanation_parts.append(f"Based on the analysis, this customer is {churn_status} with {probability:.1%} confidence.")
        
        if sorted_features:
            top_feature = sorted_features[0][0]
            top_importance = sorted_features[0][1]
            explanation_parts.append(f"The most important factor is '{top_feature}' with an importance score of {top_importance:.3f}.")
        
        if prediction == 1:
            explanation_parts.append("Consider implementing retention strategies to improve customer satisfaction and reduce churn risk.")
        else:
            explanation_parts.append("The customer shows positive indicators for continued engagement.")
        
        explanation = " ".join(explanation_parts)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "explanation": explanation,
            "model_type": "ML Model",
            "timestamp": datetime.now().isoformat(),
            "generation_method": "Rule-based"
        }
    
    def generate_batch_explanations(self, 
                                  predictions: List[int],
                                  probabilities: List[float],
                                  customer_data_list: List[Dict],
                                  feature_importance: Dict,
                                  model_type: str = "Random Forest") -> List[Dict]:
        """
        Generate explanations for multiple predictions
        
        Args:
            predictions: List of predictions
            probabilities: List of probabilities
            customer_data_list: List of customer data dictionaries
            feature_importance: Feature importance scores
            model_type: Type of ML model used
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i, (prediction, probability, customer_data) in enumerate(zip(predictions, probabilities, customer_data_list)):
            try:
                explanation = self.generate_explanation(
                    prediction, probability, customer_data, feature_importance, model_type
                )
                explanation["customer_index"] = i
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Error generating explanation for customer {i}: {str(e)}")
                # Add fallback explanation
                fallback = self._generate_simple_explanation(prediction, probability, customer_data, feature_importance)
                fallback["customer_index"] = i
                explanations.append(fallback)
        
        return explanations
    
    def generate_model_summary(self, 
                             training_results: Dict,
                             feature_importance: Dict) -> Dict:
        """
        Generate a summary explanation of the trained model
        
        Args:
            training_results: Results from model training
            feature_importance: Feature importance scores
            
        Returns:
            Model summary explanation
        """
        try:
            if not self.api_key:
                return self._generate_simple_model_summary(training_results, feature_importance)
            
            # Create summary prompt
            prompt = f"""
            Provide a business-friendly summary of a customer churn prediction model:

            MODEL PERFORMANCE:
            - Model Type: {training_results.get('model_type', 'Unknown')}
            - Accuracy: {training_results.get('accuracy', 0):.2%}
            - Training Samples: {training_results.get('training_samples', 0)}
            - Test Samples: {training_results.get('test_samples', 0)}
            - Features Used: {training_results.get('feature_count', 0)}

            TOP FEATURES (by importance):
            {chr(10).join([f"{feature}: {importance:.3f}" for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]])}

            Please provide a 2-3 sentence summary that:
            1. Explains the model's performance in business terms
            2. Highlights the most important features for churn prediction
            3. Suggests how this model can be used for business decisions

            Summary:
            """
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data science consultant explaining ML model results to business stakeholders."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                "model_summary": summary,
                "performance_metrics": training_results,
                "top_features": dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating model summary: {str(e)}")
            return self._generate_simple_model_summary(training_results, feature_importance)
    
    def _generate_simple_model_summary(self, training_results: Dict, feature_importance: Dict) -> Dict:
        """
        Generate a simple model summary without OpenAI API
        """
        accuracy = training_results.get('accuracy', 0)
        model_type = training_results.get('model_type', 'ML Model')
        
        # Get top features
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        summary = f"The {model_type} achieved {accuracy:.1%} accuracy in predicting customer churn. "
        summary += f"The most important factors for churn prediction are: {', '.join(list(top_features.keys())[:3])}. "
        summary += "This model can help identify at-risk customers and guide retention strategies."
        
        return {
            "model_summary": summary,
            "performance_metrics": training_results,
            "top_features": top_features,
            "timestamp": datetime.now().isoformat()
        } 