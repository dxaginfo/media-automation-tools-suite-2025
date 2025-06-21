import cv2
import numpy as np
import re
import json
from datetime import datetime
import os

try:
    import google.generativeai as genai
    from google.cloud import vision
    from google.cloud import storage
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False

class FrameAnalyzer:
    def __init__(self, video_path, sensitivity=0.7):
        self.video_path = video_path
        self.sensitivity = sensitivity
        
        if GOOGLE_APIS_AVAILABLE:
            self.vision_client = vision.ImageAnnotatorClient()
        
    def extract_frames(self, interval_seconds=1):
        """Extract frames at regular intervals"""
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        frames = []
        
        success, frame = video.read()
        count = 0
        
        while success:
            if count % frame_interval == 0:
                frames.append({
                    "frame": frame,
                    "timestamp": count / fps
                })
            success, frame = video.read()
            count += 1
            
        video.release()
        return frames
    
    def detect_scene_transitions(self):
        """Detect significant changes between frames indicating scene transitions"""
        frames = self.extract_frames(0.5)  # More frequent sampling for transition detection
        transitions = []
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]["frame"]
            curr_frame = frames[i]["frame"]
            
            # Convert to grayscale for histogram comparison
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Compare histograms
            prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
            curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            
            similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            
            # If frames are significantly different, mark as transition
            if similarity < (1 - self.sensitivity):
                transitions.append({
                    "timestamp": frames[i]["timestamp"],
                    "from_frame_index": i-1,
                    "to_frame_index": i,
                    "similarity_score": similarity
                })
                
        return transitions


class ScriptParser:
    def __init__(self, script_path):
        self.script_path = script_path
        
    def extract_text(self):
        """Extract text from various file formats"""
        if self.script_path.endswith('.txt'):
            with open(self.script_path, 'r') as file:
                return file.read()
        elif self.script_path.endswith('.pdf'):
            # This requires the pdf2image and pytesseract libraries
            try:
                from pdf2image import convert_from_path
                import pytesseract
                
                pages = convert_from_path(self.script_path)
                text = ""
                for page in pages:
                    text += pytesseract.image_to_string(page)
                return text
            except ImportError:
                raise ImportError("pdf2image and pytesseract are required for PDF processing")
        elif self.script_path.endswith('.docx'):
            try:
                import docx2txt
                return docx2txt.process(self.script_path)
            except ImportError:
                raise ImportError("docx2txt is required for DOCX processing")
        else:
            raise ValueError("Unsupported script format")
    
    def parse_scenes(self):
        """Extract scene information from script text"""
        text = self.extract_text()
        
        # This regex pattern looks for scene headers in standard screenplay format
        # Adjust based on your script format
        scene_pattern = r'(INT\.|EXT\.|INT/EXT\.)\s*(.*?)(?=\n|$)'
        scenes = []
        
        for match in re.finditer(scene_pattern, text, re.MULTILINE):
            scene_type = match.group(1)  # INT. or EXT.
            scene_description = match.group(2).strip()
            position = match.start()
            
            # Get the content until the next scene or end of text
            next_match = re.search(scene_pattern, text[position + len(match.group(0)):])
            if next_match:
                scene_content = text[position:position + len(match.group(0)) + next_match.start()]
            else:
                scene_content = text[position:]
            
            scenes.append({
                "type": scene_type,
                "description": scene_description,
                "content": scene_content,
                "position": position
            })
            
        return scenes


class ContinuityChecker:
    def __init__(self, api_key, model="gemini-pro-vision"):
        self.model = model
        if GOOGLE_APIS_AVAILABLE:
            genai.configure(api_key=api_key)
        else:
            raise ImportError("Google Generative AI library is required for continuity checking")
        
    def analyze_continuity(self, frame_before, frame_after, scene_context=None):
        """Use Gemini to analyze continuity between frames"""
        # Convert frames to base64 for API
        import base64
        
        _, img_before_bytes = cv2.imencode('.jpg', frame_before)
        _, img_after_bytes = cv2.imencode('.jpg', frame_after)
        
        img_before_b64 = base64.b64encode(img_before_bytes).decode('utf-8')
        img_after_b64 = base64.b64encode(img_after_bytes).decode('utf-8')
        
        # Prepare prompt with context
        prompt = """
        Analyze these two consecutive frames for continuity errors. 
        Look for discrepancies in:
        - Lighting and color grading
        - Props and their positions
        - Character positions and costumes
        - Background elements
        
        Return your analysis as a JSON object with:
        - A boolean "has_errors" field
        - An array of "issues" with "type", "severity", and "description"
        - A "confidence" score from 0-1
        """
        
        if scene_context:
            prompt += f"\nContext from script: {scene_context}"
        
        try:
            # Call Gemini API
            model = genai.GenerativeModel(self.model)
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_before_b64},
                {"mime_type": "image/jpeg", "data": img_after_b64}
            ])
            
            # Parse the response to extract the JSON
            result_text = response.text
            # Extract JSON object if embedded in text
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = {"error": "Could not extract valid JSON from response"}
        except Exception as e:
            result = {"error": str(e)}
            
        return result


class SceneValidator:
    def __init__(self, config=None):
        self.config = config or {}
        self.api_key = self.config.get("gemini_api_key")
        
    def validate(self, video_path, script_path=None):
        """Run full validation process"""
        # Initialize components
        frame_analyzer = FrameAnalyzer(
            video_path, 
            sensitivity=self.config.get("sensitivity", 0.7)
        )
        
        # Detect scene transitions
        transitions = frame_analyzer.detect_scene_transitions()
        
        # Extract frames at transitions for analysis
        video = cv2.VideoCapture(video_path)
        validation_results = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "issues_count": 0
            },
            "scene_transitions": []
        }
        
        # If script provided, parse scenes
        script_scenes = None
        if script_path:
            script_parser = ScriptParser(script_path)
            script_scenes = script_parser.parse_scenes()
        
        # Initialize continuity checker if API key is provided
        continuity_checker = None
        if self.api_key and GOOGLE_APIS_AVAILABLE:
            continuity_checker = ContinuityChecker(self.api_key)
        
        # Analyze each transition
        for i, transition in enumerate(transitions):
            # Get frames before and after transition
            video.set(cv2.CAP_PROP_POS_MSEC, transition["timestamp"] * 1000 - 500)  # 0.5s before
            success_before, frame_before = video.read()
            
            video.set(cv2.CAP_PROP_POS_MSEC, transition["timestamp"] * 1000 + 500)  # 0.5s after
            success_after, frame_after = video.read()
            
            transition_data = {
                "timestamp": self._format_timestamp(transition["timestamp"]),
                "issues": []
            }
            
            # If we have a continuity checker, use it
            if continuity_checker and success_before and success_after:
                # Get script context if available
                scene_context = None
                if script_scenes:
                    # Logic to match timestamp to script position (simplified)
                    # This would need to be more sophisticated in a real implementation
                    scene_context = script_scenes[min(i, len(script_scenes)-1)]["content"]
                
                analysis = continuity_checker.analyze_continuity(
                    frame_before, frame_after, scene_context
                )
                
                if analysis.get("has_errors", False):
                    for issue in analysis.get("issues", []):
                        transition_data["issues"].append({
                            "type": issue.get("type", "unknown"),
                            "severity": issue.get("severity", "medium"),
                            "description": issue.get("description", ""),
                            "suggestion": issue.get("suggestion", "")
                        })
            
            validation_results["scene_transitions"].append(transition_data)
            validation_results["validation_summary"]["issues_count"] += len(transition_data["issues"])
        
        validation_results["validation_summary"]["pass"] = validation_results["validation_summary"]["issues_count"] == 0
        validation_results["validation_summary"]["score"] = self._calculate_score(validation_results)
        
        video.release()
        return validation_results
    
    def _format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS.ms format"""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds_remainder = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"
    
    def _calculate_score(self, results):
        """Calculate quality score based on issues"""
        issues = results["validation_summary"]["issues_count"]
        
        # Simple scoring algorithm - can be made more sophisticated
        if issues == 0:
            return 100
        elif issues < 5:
            return 90 - (issues * 5)
        elif issues < 10:
            return 70 - ((issues - 5) * 4)
        else:
            return max(10, 50 - ((issues - 10) * 2))


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scene Validator Tool')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--script', help='Path to script file (optional)')
    parser.add_argument('--sensitivity', type=float, default=0.7, 
                        help='Sensitivity for scene detection (0-1)')
    parser.add_argument('--check-types', default='lighting,props,costume,timing',
                        help='Comma-separated list of checks to perform')
    parser.add_argument('--output-format', choices=['json', 'text'], default='json',
                        help='Format for output results')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to a file')
    parser.add_argument('--results-path', default='./validation_results/',
                        help='Path to save results')
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get('GEMINI_API_KEY')
    
    # Create config from arguments
    config = {
        "gemini_api_key": api_key,
        "sensitivity": args.sensitivity,
        "check_types": args.check_types.split(','),
        "output_format": args.output_format,
        "save_results": args.save_results,
        "results_path": args.results_path
    }
    
    # Create validator and run validation
    validator = SceneValidator(config)
    results = validator.validate(args.video, args.script)
    
    # Print or save results based on settings
    if args.output_format == 'json':
        output = json.dumps(results, indent=2)
    else:
        # Text format (simple summary)
        output = f"Scene Validation Results\n"
        output += f"=====================\n\n"
        output += f"Validation Score: {results['validation_summary']['score']}/100\n"
        output += f"Issues Found: {results['validation_summary']['issues_count']}\n"
        output += f"Timestamp: {results['validation_summary']['timestamp']}\n\n"
        
        if results['scene_transitions']:
            output += f"Scene Transitions with Issues:\n"
            for transition in results['scene_transitions']:
                if transition['issues']:
                    output += f"\nAt {transition['timestamp']}:\n"
                    for issue in transition['issues']:
                        output += f"- {issue['type']} ({issue['severity']}): {issue['description']}\n"
                        if issue.get('suggestion'):
                            output += f"  Suggestion: {issue['suggestion']}\n"
    
    if args.save_results:
        os.makedirs(args.results_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.results_path}/validation_{timestamp}"
        extension = ".json" if args.output_format == 'json' else ".txt"
        
        with open(filename + extension, 'w') as f:
            f.write(output)
        print(f"Results saved to {filename}{extension}")
    else:
        print(output)