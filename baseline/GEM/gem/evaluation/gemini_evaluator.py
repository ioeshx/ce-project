import os
from collections import defaultdict, Counter
from typing import List

from tqdm import tqdm

from google import genai

from dotenv import load_dotenv

from gem.evaluation.base_evaluator import BaseEvaluator

load_dotenv()


def get_mime_type(image_path):
    """Get MIME type from file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.heic': 'image/heic',
        '.heif': 'image/heif'
    }
    return mime_type_map.get(ext, 'image/jpeg')


class GeminiClassifier:
    """
    Wrapper for the Gemini API for prompt-based image classification.
    """

    # Default prompt template instructing the model on the task and output format
    DEFAULT_PROMPT = """
Analyze the provided image and answer the binary question below.
Your output MUST follow the following format.

{label_instruction}

QUESTION: {question}

ANSWER: """

    BINARY_LABEL_INSTRUCTION = """Answer only YES or NO. If you are unsure, answer NO."""
    MULTI_LABEL_INSTRUCTION = """Classify as one of the following labels: {labels}.
Answer only with ONE label from this list."""

    def __init__(
            self, question: str, model_name: str = 'gemini-2.5-flash',
            labels: List[str] = None, success_label: str = None
    ):
        """
        Initializes the Gemini client and prepares the classification prompt.

        :param question: The specific instructions for Gemini (e.g., "Classify this image as 'Indoor' or 'Outdoor'").
        :param labels: Optional list of labels. Defaults to ["NO", "YES"] as a binary classifier.
        """
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model_name
        self.full_prompt = self.DEFAULT_PROMPT.format(
            question=question,
            label_instruction=self.MULTI_LABEL_INSTRUCTION.format(
                labels=", ".join(labels)) if labels else self.BINARY_LABEL_INSTRUCTION
        )

        if not labels:
            self.labels = ["NO", "YES"]
            self.success_label = "YES"
            self.is_binary_classifier = True
        else:
            self.labels = labels
            self.success_label = success_label
            self.is_binary_classifier = False

        print("Created GeminiClassifier for the following prompt:")
        print("> Question:", question)
        print("> Possible Labels:", self.labels)

    def __call__(self, image_path: str):
        """
        Sends the image and prompt to the Gemini API for classification.
        """
        try:
            # 1. Load the image using PIL
            # Load the image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            mime_type = get_mime_type(image_path)

            # 2. Call the Gemini API with multimodal input (text + image)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[self.full_prompt, genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
            )

            # 3. Parse the JSON response
            # The model is instructed to return *only* the JSON object
            text = response.text.lower()

            if "yes" in text:
                label = "YES"
            else:
                label = "NO"

            return {
                "label": label,
                "success": label == self.success_label,
                "error": None,
                "response": text
            }

        except Exception as e:
            print(e)
            return {
                "labels": "",
                "success": False,
                "error": str(e),
                "response": None
            }


class GeminiEvaluator(BaseEvaluator):
    """
    Evaluates a set of images using the Gemini API based on a custom prompt.
    """

    def __init__(self, *args, **kwargs):
        super(GeminiEvaluator, self).__init__()

        # Create the Gemini classifier
        self.classifier = GeminiClassifier(*args, **kwargs)

    def process_images(self, image_file_paths, **kwargs) -> dict:

        # Initialize counters
        summary = Counter()
        label_counter = defaultdict(int)

        print(f"Scanning {len(image_file_paths)} images using prompt:\n{self.classifier.full_prompt}\n")

        total_images = len(image_file_paths)

        for label in self.classifier.labels:
            label_counter[label] = 0

        for image_path in tqdm(image_file_paths, desc="Processing images with Gemini"):
            result = self.classifier(image_path)

            print("Gemini Response:", result.get('response', ''))

            if result["error"]:
                print("Error:", result['error'])
                summary['errors'] += 1
            else:
                summary['no_errors'] += 1
                label = result['label']
                label_counter[label] += 1

                if result['success']:
                    summary['successes'] += 1

        # Final summary
        total = summary['no_errors'] + summary['errors']

        n_successes = summary.get("successes", 0)
        detection_rate = (n_successes / total) if total > 0 else 0.0

        return {
            'total': total_images,
            'processed': total,
            'errors': summary['errors'],
            'detection_rate': detection_rate,
            **dict(label_counter)
        }
