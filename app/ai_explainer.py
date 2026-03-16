"""
ai_explainer.py - AI-Powered Root Cause Analysis using Free LLMs

FIXES:
- Added validate_key() method for instant key testing
- Rate limit protection for batch mode
- Proper error handling for all Gemini response types

Author: BCA Final Year Project
Date: 2026
"""

import os
import time
from app.utils import get_display_name, ANOMALY_DESCRIPTIONS


class AIExplainer:
    """
    Generates human-readable explanations for anomaly predictions.
    """
    
    def __init__(self, provider='rule_based', api_key=None):
        self.provider = provider
        self.manual_key = api_key
        self.active_key = None
        self.using_system_key = False
        self.llm_available = False
        self.model = None
        self.error_message = ""
        self.validated = False
        self.validation_attempted = False
        self.rate_limited = False
        self.last_call_time = 0
        self.min_call_interval = 2.0
        
        # Resolve which key to use
        self._resolve_api_key()
        self._setup_llm()

    def _resolve_api_key(self):
        """
        Determine which API key to use.
        
        Priority:
        1. Manual key (user typed in sidebar)
        2. System key (from .streamlit/secrets.toml or environment)
        3. No key → rule-based
        """
        # Priority 1: User's manual key
        if self.manual_key and len(self.manual_key.strip()) > 10:
            self.active_key = self.manual_key.strip()
            self.using_system_key = False
            return
        
        # Priority 2: System key
        system_key = self._get_system_key()
        if system_key:
            self.active_key = system_key
            self.using_system_key = True
            return
        
        # Priority 3: No key
        self.active_key = None

    def _get_system_key(self):
        """
        Get API key from environment.
        
        Checks:
        1. Streamlit secrets (.streamlit/secrets.toml)
        2. OS environment variable
        """
        import streamlit as st
        
        key = None
        
        # Try Streamlit secrets
        try:
            if self.provider == 'groq':
                key = st.secrets.get("GROQ_API_KEY", None)
            elif self.provider == 'gemini':
                key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            pass
        
        # Try OS environment variable
        if not key:
            if self.provider == 'groq':
                key = os.environ.get("GROQ_API_KEY", None)
            elif self.provider == 'gemini':
                key = os.environ.get("GEMINI_API_KEY", None)
        
        if key and len(str(key).strip()) > 10:
            return str(key).strip()
        
        return None
    
        # REPLACE WITH THIS:
    
    def _setup_llm(self):
        """Setup LLM without making a test call."""
        self.llm_available = False
        self.error_message = ""
        self.validated = False
        self.validation_attempted = False
        self.rate_limited = False
        
        if self.provider == 'rule_based':
            return
        
        if not self.active_key:                                       # ← FIXED
            self.error_message = "No API key available"
            return
        
        if self.provider == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.active_key)              # ← FIXED
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.llm_available = True
            except ImportError:
                self.error_message = "Run: pip install google-generativeai"
            except Exception as e:
                self.error_message = f"Setup error: {str(e)[:150]}"
        
        elif self.provider == 'groq':
            try:
                from groq import Groq
                self.model = Groq(api_key=self.active_key)            # ← FIXED
                self.llm_available = True
            except ImportError:
                self.error_message = "Run: pip install groq"
            except Exception as e:
                self.error_message = f"Setup error: {str(e)[:150]}"
    
    # ← NEW: Separate method to validate key on demand
    def validate_key(self):
        """
        Validate the API key by making a single lightweight test call.
        
        WHY SEPARATE METHOD?
        - Called only when user clicks "Test Connection"
        - Not called during __init__ (would slow down startup)
        - Not called during batch processing (would waste quota)
        - One clean test → clear yes/no answer
        
        Returns:
            dict with 'valid' (bool), 'message' (str)
        """
        if self.provider == 'rule_based':
            return {'valid': True, 'message': 'Rule-based mode (no API needed)'}
        
        if not self.model:
            return {'valid': False, 'message': self.error_message or 'Model not initialized'}
        
        try:
            if self.provider == 'gemini':
                response = self.model.generate_content("Say OK")
                
                # Handle different response structures
                text = ""
                if hasattr(response, 'text'):
                    text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    try:
                        text = response.candidates[0].content.parts[0].text
                    except (IndexError, AttributeError):
                        text = "response received"
                
                if text:
                    self.validated = True
                    self.validation_attempted = True
                    self.llm_available = True
                    self.error_message = ""
                    return {'valid': True, 'message': 'Connected to Google Gemini ✅'}
                else:
                    self.validated = False
                    self.validation_attempted = True
                    self.error_message = "Empty response from API"
                    return {'valid': False, 'message': 'API returned empty response'}
            
            elif self.provider == 'groq':
                response = self.model.chat.completions.create(
                    messages=[{"role": "user", "content": "Say OK"}],
                    model="llama-3.1-8b-instant",
                    max_tokens=5
                )
                
                if response and response.choices:
                    self.validated = True
                    self.validation_attempted = True
                    self.llm_available = True
                    self.error_message = ""
                    return {'valid': True, 'message': 'Connected to Groq ✅'}
                else:
                    self.validated = False
                    self.validation_attempted = True
                    return {'valid': False, 'message': 'Empty response'}
        
        except Exception as e:
            self.validated = False
            self.validation_attempted = True
            self.llm_available = False
            
            error_str = str(e).lower()
            
            if 'api_key' in error_str or 'invalid' in error_str or 'authenticate' in error_str:
                self.error_message = "Invalid API key"
            elif 'quota' in error_str or 'rate' in error_str or '429' in error_str:
                self.error_message = "Rate limit hit. Wait 60 seconds and try again."
            elif 'permission' in error_str or 'forbidden' in error_str or '403' in error_str:
                self.error_message = "API key doesn't have required permissions"
            elif 'not found' in error_str or '404' in error_str:
                self.error_message = "Model not found. Check API configuration."
            else:
                self.error_message = f"Error: {str(e)[:100]}"
            
            return {'valid': False, 'message': self.error_message}
    
    def explain(self, prediction_result, event_data=None, use_llm=True):
        """Generate explanation for a prediction."""
        if not prediction_result['is_anomaly']:
            return {
                'explanation': '✅ All system metrics are within normal operating ranges.',
                'root_cause': 'N/A - System operating normally',
                'impact': 'No impact - Normal operation',
                'recommendation': 'Continue monitoring. No action required.',
                'provider': 'rule_based',
                'llm_error': ''                                        # ← NEW
            }
        
        if not use_llm:
            result = self._rule_based_explain(prediction_result, event_data)
            result['llm_error'] = 'Batch mode - rule-based used to save API quota'
            return result
        
        if self.rate_limited:
            result = self._rule_based_explain(prediction_result, event_data)
            result['llm_error'] = 'Rate limit reached - using rule-based'
            return result
        
        if self.llm_available and self.model is not None:
            time_since_last = time.time() - self.last_call_time
            
            if time_since_last < self.min_call_interval:
                result = self._rule_based_explain(prediction_result, event_data)
                result['llm_error'] = f'Rate protection - wait {self.min_call_interval}s between calls'
                return result
            
            try:
                result = self._llm_explain(prediction_result, event_data)
                self.validated = True
                self.validation_attempted = True
                self.error_message = ""
                self.rate_limited = False
                self.last_call_time = time.time()
                result['llm_error'] = ''                               # ← No error
                return result
            except Exception as e:
                self.validation_attempted = True
                error_str = str(e).lower()
                
                if 'api_key' in error_str or 'invalid' in error_str or 'auth' in error_str:
                    self.error_message = "Invalid API key"
                    self.llm_available = False
                    self.validated = False
                elif 'quota' in error_str or 'rate' in error_str or '429' in error_str:
                    self.error_message = "Rate limit reached"
                    self.rate_limited = True
                elif 'blocked' in error_str or 'safety' in error_str:
                    self.error_message = "Content filtered by safety"
                else:
                    self.error_message = f"LLM error: {str(e)[:100]}"
                
                result = self._rule_based_explain(prediction_result, event_data)
                result['llm_error'] = self.error_message               # ← Track WHY it failed
                return result
        else:
            result = self._rule_based_explain(prediction_result, event_data)
            result['llm_error'] = self.error_message or 'LLM not available'
            return result
    
    def _build_prompt(self, prediction_result, event_data=None):
        """Build prompt for the LLM."""
        severity = prediction_result['severity']
        score_pct = prediction_result['anomaly_score_pct']
        contributions = prediction_result.get('contributions', [])[:5]
        anomaly_type = event_data.get('anomaly_type', 'unknown') if event_data else 'unknown'
        
        feature_details = ""
        for feat in contributions:
            feature_details += (
                f"  - {feat['feature_display']}: "
                f"z-score={feat['z_score']} std devs ({feat['direction']}), "
                f"value={feat['raw_display']}\n"
            )
        
        prompt = f"""You are a mobile app performance expert. Analyze this anomaly briefly.

Severity: {severity} | Score: {score_pct:.0f}% | Pattern: {anomaly_type}

Key metrics:
{feature_details}
Respond in this exact format (plain text only, no markdown, no asterisks):

ROOT CAUSE: One sentence about the likely cause.
IMPACT: One sentence about user impact.
RECOMMENDATION: Two to three specific actions to fix this."""
        
        return prompt
    
    def _llm_explain(self, prediction_result, event_data=None):
        """Call the LLM provider."""
        prompt = self._build_prompt(prediction_result, event_data)
        text = ""
        
        if self.provider == 'gemini':
            response = self.model.generate_content(prompt)
            
            if hasattr(response, 'text') and response.text:
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    text = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    raise ValueError("Could not extract text from Gemini response")
            else:
                raise ValueError("Gemini returned empty response")
        
        elif self.provider == 'groq':
            response = self.model.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=250
            )
            if response and response.choices:
                text = response.choices[0].message.content
            else:
                raise ValueError("Groq returned empty response")
        
        if not text or len(text.strip()) < 20:
            raise ValueError("Response too short")
        
        return self._parse_llm_response(text)
    
    def _parse_llm_response(self, text):
        """Parse LLM response into structured sections."""
        result = {
            'explanation': text,
            'root_cause': '',
            'impact': '',
            'recommendation': '',
            'provider': self.provider
        }
        
        lines = text.strip().split('\n')
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            if line_upper.startswith('ROOT CAUSE:'):
                result['root_cause'] = line_clean.split(':', 1)[1].strip()
            elif line_upper.startswith('IMPACT:'):
                result['impact'] = line_clean.split(':', 1)[1].strip()
            elif 'RECOMMENDATION' in line_upper and ':' in line_clean:
                rec_lines = [line_clean.split(':', 1)[1].strip()]
                for remaining in lines[i+1:]:
                    remaining = remaining.strip()
                    if remaining and not remaining.upper().startswith(('ROOT', 'IMPACT')):
                        rec_lines.append(remaining)
                result['recommendation'] = ' '.join(rec_lines)
        
        if not result['root_cause']:
            result['root_cause'] = text[:200]
        if not result['recommendation']:
            result['recommendation'] = 'Investigate the flagged metrics.'
        
        return result
    
    def _rule_based_explain(self, prediction_result, event_data=None):
        """Generate explanation using rules (no LLM needed)."""
        severity = prediction_result['severity']
        contributions = prediction_result.get('contributions', [])
        
        top_features = contributions[:3] if contributions else []
        feature_names = [f['feature'].lower() for f in top_features]
        feature_names_str = ' '.join(feature_names)
        
        if 'memory' in feature_names_str or 'memory_growth' in feature_names_str:
            root_cause = (
                f"Memory usage has spiked to abnormal levels "
                f"({top_features[0]['raw_display'] if top_features else 'N/A'}). "
                f"This pattern is consistent with a memory leak."
            )
            impact = "Users may experience app slowdowns, UI freezes, and potential crashes."
            recommendation = (
                "1) Check for unclosed database connections or file handles. "
                "2) Review recent code changes for objects not being garbage collected. "
                "3) Use memory profiling tools to identify the leaking component. "
                "4) Monitor if memory returns to normal after session ends."
            )
        elif 'latency' in feature_names_str or 'api' in feature_names_str:
            root_cause = (
                f"API response times have increased significantly "
                f"({top_features[0]['raw_display'] if top_features else 'N/A'}). "
                f"Backend performance degradation detected."
            )
            impact = "Users will experience slow screen loads and timeout errors."
            recommendation = (
                "1) Check backend server CPU and memory utilization. "
                "2) Review slow query logs in the database. "
                "3) Verify CDN and network infrastructure health. "
                "4) Check if recent deployment introduced regression."
            )
        elif 'fps' in feature_names_str:
            root_cause = (
                f"Frame rate has dropped below acceptable levels "
                f"({top_features[0]['raw_display'] if top_features else 'N/A'}). "
                f"Heavy UI rendering or main thread blocking detected."
            )
            impact = "Users experience visible UI jank and sluggish touch responses."
            recommendation = (
                "1) Profile the UI thread for blocking operations. "
                "2) Move heavy computations to background threads. "
                "3) Optimize list rendering and image loading. "
                "4) Check for excessive redraws."
            )
        elif 'error' in feature_names_str:
            root_cause = (
                f"Error rate has spiked abnormally "
                f"({top_features[0]['raw_display'] if top_features else 'N/A'}). "
                f"Systematic failure detected."
            )
            impact = "Users encountering frequent errors, unable to complete app flows."
            recommendation = (
                "1) Review error logs for common error codes. "
                "2) Check if a specific API endpoint is failing. "
                "3) Verify third-party service health. "
                "4) Roll back recent deployments if error spike coincides."
            )
        else:
            root_cause = (
                f"Multiple metrics deviating from normal. "
                f"Top factor: {top_features[0]['feature_display'] if top_features else 'unknown'} "
                f"(z-score: {top_features[0]['z_score'] if top_features else 'N/A'}σ)."
            )
            impact = "Combined deviations may affect multiple aspects of user experience."
            recommendation = (
                "1) Review system-wide health dashboards. "
                "2) Check for infrastructure issues. "
                "3) Correlate with recent deployments. "
                "4) Monitor if anomaly persists."
            )
        
        return {
            'explanation': f"🔍 **{severity} Anomaly**\n\n**Root Cause:** {root_cause}\n\n**Impact:** {impact}\n\n**Actions:** {recommendation}",
            'root_cause': root_cause,
            'impact': impact,
            'recommendation': recommendation,
            'provider': 'rule_based',
            'llm_error': ''
        }
    
    def get_status(self):
        """Return current provider status."""
        if self.provider == 'rule_based':
            return {
                'provider': 'rule_based',
                'llm_available': False,
                'validated': False,
                'status': '📝 Rule-Based (No API Key Needed)',
                'error': ''
            }
        
        if self.validated:
            provider_name = 'Google Gemini' if self.provider == 'gemini' else 'Groq'
            return {
                'provider': self.provider,
                'llm_available': True,
                'validated': True,
                'status': f'✅ Connected to {provider_name}',
                'error': ''
            }
        elif self.llm_available and not self.validation_attempted:
            provider_name = 'Google Gemini' if self.provider == 'gemini' else 'Groq'
            return {
                'provider': self.provider,
                'llm_available': True,
                'validated': False,
                'status': f'🔄 {provider_name} — Key entered, click Test to verify',
                'error': ''
            }
        elif self.validation_attempted and not self.validated:
            return {
                'provider': self.provider,
                'llm_available': False,
                'validated': False,
                'status': '❌ Connection Failed',
                'error': self.error_message
            }
        else:
            return {
                'provider': self.provider,
                'llm_available': False,
                'validated': False,
                'status': '⚠️ Enter API key',
                'error': self.error_message
            }