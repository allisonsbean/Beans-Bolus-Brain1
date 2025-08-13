import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import streamlit as st

class GlucosePredictionEngine:
    """
    Advanced glucose prediction system for Bean's Bolus Brain
    Predicts glucose levels 1-2 hours ahead using trend analysis and IOB modeling
    """
    
    def __init__(self):
        self.prediction_windows = [60, 120]  # 1 and 2 hour predictions in minutes
        self.min_readings_for_prediction = 3
        self.max_reading_gap_minutes = 15  # Max gap between readings to maintain trend
        self.max_realistic_rate = 2.0  # Max realistic glucose change: 2 mg/dL per minute
        self.max_prediction_value = 350  # Cap predictions at 350 mg/dL
        self.min_prediction_value = 40   # Floor predictions at 40 mg/dL
        
    def predict_glucose_trends(self, glucose_readings: List[Dict], current_iob: float, 
                              recent_meals: List[Dict] = None) -> Dict:
        """
        Main prediction function that analyzes trends and generates alerts
        """
        if len(glucose_readings) < self.min_readings_for_prediction:
            return self._no_prediction_response("Insufficient data for prediction")
            
        # Sort readings by timestamp (most recent first)
        sorted_readings = sorted(glucose_readings, 
                               key=lambda x: x['timestamp'], reverse=True)
        
        # Filter to only use Dexcom readings for trend analysis
        dexcom_readings = [r for r in sorted_readings if r.get('trend', '') != 'Manual Entry']
        
        # Need at least 2 Dexcom readings for trend
        if len(dexcom_readings) < 2:
            return self._no_prediction_response("Need at least 2 Dexcom readings for reliable trends")
        
        # Check for data freshness and gaps
        data_quality = self._assess_data_quality(dexcom_readings)
        if not data_quality['usable']:
            return self._no_prediction_response(data_quality['reason'])
            
        # Calculate glucose trend using only Dexcom data
        trend_analysis = self._calculate_glucose_trend(dexcom_readings[:4])
        
        # Factor in IOB impact
        iob_impact = self._calculate_iob_impact(current_iob)
        
        # Use most recent reading (even if manual) as starting point
        current_glucose = sorted_readings[0]['value']
        
        # Generate predictions for 1 and 2 hour windows
        predictions = {}
        alerts = []
        
        for window_minutes in self.prediction_windows:
            prediction = self._predict_glucose_at_time(
                current_glucose=current_glucose,
                trend_rate=trend_analysis['rate_per_minute'],
                iob_impact=iob_impact,
                minutes_ahead=window_minutes,
                trend_confidence=trend_analysis['confidence']
            )
            
            window_hours = window_minutes // 60
            predictions[f'{window_hours}h'] = prediction
            
            # Check for alerts
            alert = self._check_for_alerts(prediction, window_hours)
            if alert:
                alerts.append(alert)
        
        return {
            'predictions': predictions,
            'alerts': alerts,
            'trend_analysis': trend_analysis,
            'iob_impact': iob_impact,
            'data_quality': data_quality
        }
    
    def _assess_data_quality(self, readings: List[Dict]) -> Dict:
        """Assess if glucose data is suitable for prediction"""
        if not readings:
            return {'usable': False, 'reason': 'No Dexcom readings available'}
            
        # Check if most recent reading is too old
        most_recent = readings[0]['timestamp']
        if isinstance(most_recent, str):
            most_recent = datetime.fromisoformat(most_recent.replace('Z', '+00:00'))
        
        minutes_old = (datetime.now().replace(tzinfo=most_recent.tzinfo) - most_recent).total_seconds() / 60
        
        if minutes_old > 60:  # More lenient for Dexcom-only data
            return {'usable': False, 'reason': f'Most recent Dexcom reading is {minutes_old:.0f} minutes old'}
            
        return {
            'usable': True,
            'confidence': 0.8,
            'minutes_old': minutes_old,
            'dexcom_readings': len(readings)
        }
    
    def _calculate_glucose_trend(self, readings: List[Dict]) -> Dict:
        """Calculate glucose trend rate using only Dexcom readings"""
        if len(readings) < 2:
            return {'rate_per_minute': 0, 'confidence': 0, 'direction': 'stable'}
            
        # Use linear regression on Dexcom readings for better trend
        times = []
        values = []
        
        for reading in readings:
            timestamp = reading['timestamp']
            if isinstance(timestamp, str):
                # Convert string timestamp to datetime for calculation
                timestamp = datetime.strptime(timestamp, '%m/%d %H:%M')
            
            # Convert to minutes since first reading
            if not times:
                base_time = timestamp
                times.append(0)
            else:
                time_diff = (timestamp - base_time).total_seconds() / 60
                times.append(-time_diff)  # Negative because we're going backward in time
            
            values.append(reading['value'])
        
        # Simple linear regression
        if len(values) >= 3:
            # Use numpy-style calculation
            times_array = times
            values_array = values
            
            # Calculate slope (trend rate)
            n = len(times_array)
            sum_x = sum(times_array)
            sum_y = sum(values_array)
            sum_xy = sum(x * y for x, y in zip(times_array, values_array))
            sum_x2 = sum(x * x for x in times_array)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 0.001:  # Avoid division by zero
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Slope is in mg/dL per minute (negative time direction)
            rate_per_minute = -slope  # Flip sign for forward time
            
            # Calculate R-squared for confidence
            y_mean = sum_y / n
            ss_tot = sum((y - y_mean) ** 2 for y in values_array)
            y_pred = [sum_y/n + slope * (x - sum_x/n) for x in times_array]
            ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(values_array, y_pred))
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                confidence = max(0.3, min(0.9, r_squared))
            else:
                confidence = 0.5
        else:
            # Simple two-point calculation
            time_diff_minutes = abs((readings[0]['timestamp'] - readings[1]['timestamp']).total_seconds() / 60) if hasattr(readings[0]['timestamp'], 'total_seconds') else 5
            if time_diff_minutes > 0:
                rate_per_minute = (values[0] - values[1]) / time_diff_minutes
                confidence = 0.6
            else:
                rate_per_minute = 0
                confidence = 0.3
        
        # Apply safety limits
        rate_per_minute = max(-self.max_realistic_rate, 
                             min(self.max_realistic_rate, rate_per_minute))
        
        # Determine direction
        if rate_per_minute > 0.5:
            direction = 'rising'
        elif rate_per_minute < -0.5:
            direction = 'falling'
        else:
            direction = 'stable'
            
        return {
            'rate_per_minute': rate_per_minute,
            'confidence': confidence,
            'direction': direction,
            'readings_used': len(values),
            'data_source': 'Dexcom only'
        }
    
    def _calculate_iob_impact(self, current_iob: float) -> Dict:
        """Calculate expected glucose impact from current IOB"""
        correction_factor = 50
        # More conservative IOB impact - spread over 3 hours instead of 2
        impact_rate_per_minute = (current_iob * correction_factor) / 180
        
        return {
            'total_expected_drop': current_iob * correction_factor,
            'impact_rate_per_minute': impact_rate_per_minute,
            'current_iob': current_iob
        }
    
    def _predict_glucose_at_time(self, current_glucose: float, trend_rate: float, 
                                iob_impact: Dict, minutes_ahead: int, trend_confidence: float) -> Dict:
        """Predict glucose value at specific time with safety limits"""
        
        # Base prediction from trend
        trend_prediction = current_glucose + (trend_rate * minutes_ahead)
        
        # Subtract IOB impact
        iob_drop = iob_impact['impact_rate_per_minute'] * minutes_ahead
        final_prediction = trend_prediction - iob_drop
        
        # Apply safety limits
        final_prediction = max(self.min_prediction_value, 
                              min(self.max_prediction_value, final_prediction))
        
        # Calculate confidence (reduce for longer predictions)
        time_decay = max(0.4, 1.0 - (minutes_ahead / 300))  # Decay over 5 hours
        final_confidence = trend_confidence * time_decay
        
        return {
            'predicted_value': round(final_prediction, 1),
            'confidence': round(final_confidence * 100, 1),
            'range_low': round(final_prediction - 15, 1),
            'range_high': round(final_prediction + 15, 1),
            'trend_component': round(trend_prediction, 1),
            'iob_component': round(-iob_drop, 1)
        }
    
    def _check_for_alerts(self, prediction: Dict, window_hours: int) -> Optional[Dict]:
        """Check if prediction triggers any alerts"""
        predicted_value = prediction['predicted_value']
        confidence = prediction['confidence']
        
        if confidence < 50:  # Higher confidence threshold
            return None
            
        alert = None
        
        # Low glucose alert
        if predicted_value < 80:
            severity = 'URGENT' if predicted_value < 70 else 'WARNING'
            alert = {
                'type': 'LOW',
                'severity': severity,
                'message': f'🔴 Predicted LOW in {window_hours}h: {predicted_value:.0f} mg/dL',
                'predicted_value': predicted_value,
                'window_hours': window_hours,
                'confidence': confidence,
                'recommendation': self._get_low_recommendation(predicted_value, window_hours)
            }
            
        # High glucose alert
        elif predicted_value > 180:
            severity = 'URGENT' if predicted_value > 250 else 'WARNING'
            alert = {
                'type': 'HIGH',
                'severity': severity,
                'message': f'🟠 Predicted HIGH in {window_hours}h: {predicted_value:.0f} mg/dL',
                'predicted_value': predicted_value,
                'window_hours': window_hours,
                'confidence': confidence,
                'recommendation': self._get_high_recommendation(predicted_value, window_hours)
            }
            
        return alert

    def _get_low_recommendation(self, predicted_value: float, window_hours: int) -> str:
        """Get recommendation for predicted low"""
        if predicted_value < 70:
            return "Consider 15-20g fast carbs now to prevent severe low"
        else:
            return f"Consider 10-15g carbs in next {max(1, window_hours-1)} hour(s)"

    def _get_high_recommendation(self, predicted_value: float, window_hours: int) -> str:
        """Get recommendation for predicted high"""
        if predicted_value > 250:
            return "Consider correction bolus now - check ketones if needed"
        else:
            return f"Monitor closely - may need correction in {window_hours} hour(s)"

    def _no_prediction_response(self, reason: str) -> Dict:
        """Return response when prediction cannot be made"""
        return {
            'predictions': {},
            'alerts': [],
            'trend_analysis': {'rate_per_minute': 0, 'confidence': 0, 'direction': 'unknown'},
            'iob_impact': {'total_expected_drop': 0, 'impact_rate_per_minute': 0, 'current_iob': 0},
            'data_quality': {'usable': False, 'reason': reason}
        }
