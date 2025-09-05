#!/usr/bin/env python3
"""
Comprehensive test suite for fine-tuned Qwen 4B intent classification model
Tests the vLLM-served model via OpenAI-compatible API
"""
import json
import time
import requests
import urllib3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dataclass
class ModelConfig:
    """Configuration for the hosted model"""
    api_endpoint: str = "model-inference-url
    default_temperature: float = 0.23
    default_max_tokens: int = 32768
    admin_username: str = "admin"
    admin_password: str = "minad"
    max_context_limit: int = 32768
    verify_ssl: bool = False
    
    # Token Authentication
    api_token: str = "access-token"
    use_token_auth: bool = True


class IntentClassifierTester:
    """Test suite for the fine-tuned intent classification model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = f"{config.api_endpoint}/v1"
        self.session = requests.Session()
        self.session.verify = config.verify_ssl
        
        # Set up authentication headers
        if config.use_token_auth:
            self.session.headers.update({
                "Authorization": f"Bearer {config.api_token}",
                "Content-Type": "application/json"
            })
        
        # Test results storage
        self.test_results = []
        
    def _create_intent_prompt(self, user_query: str) -> str:
        """Create the properly formatted prompt for intent classification"""
        return f"""<|im_start|>system
Classify the user question into one of the predefined intents. Respond with only the intent name.<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""

    def _call_model(self, prompt: str, temperature: float = None, max_tokens: int = 50) -> Tuple[str, float, bool]:
        """
        Call the vLLM model via OpenAI-compatible API
        Returns: (response_text, response_time, success)
        """
        if temperature is None:
            temperature = self.config.default_temperature
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["<|im_end|>", "\n\n"]
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content'].strip()
                    return content, response_time, True
                else:
                    return f"No choices in response: {result}", response_time, False
            else:
                return f"HTTP {response.status_code}: {response.text}", response_time, False
                
        except Exception as e:
            response_time = time.time() - start_time
            return f"Error: {str(e)}", response_time, False

    def test_basic_connectivity(self) -> bool:
        """Test if the model endpoint is accessible"""
        print("🔌 Testing basic connectivity...")
        
        try:
            # Test health endpoint
            health_response = self.session.get(f"{self.config.api_endpoint}/health", timeout=10)
            if health_response.status_code == 200:
                print("✅ Health endpoint accessible")
            else:
                print(f"⚠️  Health endpoint returned {health_response.status_code}")
            
            # Test models endpoint
            models_response = self.session.get(f"{self.base_url}/models", timeout=10)
            if models_response.status_code == 200:
                models = models_response.json()
                print(f"✅ Models endpoint accessible, found {len(models.get('data', []))} models")
                return True
            else:
                print(f"❌ Models endpoint failed: {models_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connectivity test failed: {e}")
            return False

    def test_intent_classification_accuracy(self) -> Dict:
        """Test the model's accuracy on known intent classification examples"""
        print("\n🎯 Testing intent classification accuracy...")
        
        # Test cases based on your training data
        test_cases = [
            ("follow up on ticket ?", "ComplaintStatusInquiry"),
            ("smiles faq please ?", "RetrieveSmilesInformation"),
            ("how do i update to prepaid account ?", "Migration"),
            ("What is SWYP and what's the price?", "SWYPCostFAQ"),
            ("just wanna know when my contract ends ?", "CurrentPlanActiveProdsSrvcsInquiry"),
            ("would be able to direct me to the service provider stores in my area?", "ClosestServiceProviderStoreFAQ"),
            ("i want to change my sim ?", "SimManagement"),
            ("How do I check my balance?", "AvailableBalanceRequest"),
            ("What are my current charges?", "RetrieveThirdPartyBilling"),
            ("I need help with my internet connection", "TechnicalSupport"),
            ("Can you tell me about the latest offers?", "PromotionsInquiry"),
            ("How to activate roaming service?", "RoamingServiceActivation")
        ]
        
        correct_predictions = 0
        total_tests = len(test_cases)
        results = []
        
        for i, (query, expected_intent) in enumerate(test_cases, 1):
            prompt = self._create_intent_prompt(query)
            response, response_time, success = self._call_model(prompt, temperature=self.config.default_temperature)
            
            if success:
                predicted_intent = response.strip()
                is_correct = predicted_intent.lower() == expected_intent.lower()
                if is_correct:
                    correct_predictions += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} Test {i}: '{query}'")
                print(f"      Expected: {expected_intent}")
                print(f"      Predicted: {predicted_intent}")
                print(f"      Response time: {response_time:.3f}s")
                
                results.append({
                    "query": query,
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "correct": is_correct,
                    "response_time": response_time
                })
            else:
                print(f"  ❌ Test {i} failed: {response}")
                results.append({
                    "query": query,
                    "expected": expected_intent,
                    "predicted": None,
                    "correct": False,
                    "response_time": response_time,
                    "error": response
                })
        
        accuracy = (correct_predictions / total_tests) * 100
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        
        print(f"\n📊 Accuracy Results:")
        print(f"    Correct: {correct_predictions}/{total_tests}")
        print(f"    Accuracy: {accuracy:.1f}%")
        print(f"    Average response time: {avg_response_time:.3f}s")
        
        return {
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_tests,
            "avg_response_time": avg_response_time,
            "results": results
        }

    def test_performance_benchmarks(self) -> Dict:
        """Test model performance under different conditions"""
        print("\n⚡ Testing performance benchmarks...")
        
        # Test different prompt lengths
        short_prompt = "balance check?"
        medium_prompt = "I would like to know more information about my current account balance and recent transactions."
        long_prompt = "I have been having issues with my account for the past few weeks and I need to understand what is happening with my balance, recent charges, and also want to know about any promotional offers that might be available for my account type."
        
        test_prompts = [
            ("Short prompt", short_prompt),
            ("Medium prompt", medium_prompt),
            ("Long prompt", long_prompt)
        ]
        
        results = {}
        
        for test_name, query in test_prompts:
            print(f"  Testing {test_name}...")
            prompt = self._create_intent_prompt(query)
            
            # Run multiple iterations for average
            times = []
            responses = []
            
            for _ in range(3):
                response, response_time, success = self._call_model(prompt, temperature=0.1)
                if success:
                    times.append(response_time)
                    responses.append(response)
            
            if times:
                avg_time = sum(times) / len(times)
                results[test_name] = {
                    "avg_response_time": avg_time,
                    "min_time": min(times),
                    "max_time": max(times),
                    "sample_response": responses[0] if responses else None
                }
                print(f"    Avg: {avg_time:.3f}s, Min: {min(times):.3f}s, Max: {max(times):.3f}s")
            else:
                results[test_name] = {"error": "All requests failed"}
                print(f"    ❌ All requests failed")
        
        return results

    def test_edge_cases(self) -> Dict:
        """Test model behavior on edge cases"""
        print("\n🔍 Testing edge cases...")
        
        edge_cases = [
            ("Empty query", ""),
            ("Very short", "hi"),
            ("Non-English", "مرحبا، كيف يمكنني مساعدتك؟"),
            ("Numbers only", "123456789"),
            ("Special characters", "!@#$%^&*()"),
            ("Mixed case", "HeLLo WoRLd"),
            ("Very long query", "a" * 500),
            ("SQL injection attempt", "'; DROP TABLE users; --"),
            ("Nonsensical", "asdfghjkl qwerty zxcvbn")
        ]
        
        results = []
        
        for test_name, query in edge_cases:
            print(f"  Testing {test_name}...")
            prompt = self._create_intent_prompt(query)
            response, response_time, success = self._call_model(prompt, temperature=self.config.default_temperature)
            
            result = {
                "test_name": test_name,
                "query": query,
                "response": response,
                "success": success,
                "response_time": response_time
            }
            results.append(result)
            
            status = "✅" if success else "❌"
            print(f"    {status} Response: {response[:100]}...")
        
        return results

    def run_comprehensive_test_suite(self) -> Dict:
        """Run all tests and generate a comprehensive report"""
        print("🚀 Starting comprehensive test suite for Qwen 4B Intent Classifier")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Test connectivity
        connectivity_ok = self.test_basic_connectivity()
        if not connectivity_ok:
            return {"error": "Failed connectivity test", "timestamp": start_time.isoformat()}
        
        # Run all test suites
        accuracy_results = self.test_intent_classification_accuracy()
        performance_results = self.test_performance_benchmarks()
        edge_case_results = self.test_edge_cases()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile final report
        report = {
            "test_suite": "Qwen 4B Intent Classifier",
            "model_endpoint": self.config.api_endpoint,
            "model_name": self.config.model_name,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "connectivity": connectivity_ok,
            "accuracy": accuracy_results,
            "performance": performance_results,
            "edge_cases": edge_case_results,
            "summary": {
                "overall_accuracy": accuracy_results.get("accuracy", 0),
                "avg_response_time": accuracy_results.get("avg_response_time", 0),
                "total_tests_run": len(accuracy_results.get("results", [])) + len(edge_case_results)
            }
        }
        
        print(f"\n📋 Test Suite Summary:")
        print(f"    Duration: {duration:.1f} seconds")
        print(f"    Overall Accuracy: {report['summary']['overall_accuracy']:.1f}%")
        print(f"    Avg Response Time: {report['summary']['avg_response_time']:.3f}s")
        print(f"    Temperature: {self.config.default_temperature}")
        print(f"    Total Tests: {report['summary']['total_tests_run']}")
        
        return report

    def save_report(self, report: Dict, filename: str = None):
        """Save test report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {filename}")


def main():
    """Run the test suite"""
    config = ModelConfig()
    tester = IntentClassifierTester(config)
    
    # Run comprehensive tests
    report = tester.run_comprehensive_test_suite()
    
    # Save results
    tester.save_report(report)
    
    return report


if __name__ == "__main__":
    main()
