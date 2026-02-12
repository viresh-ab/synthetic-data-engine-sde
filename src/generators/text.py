"""
Text Data Generator

Generates synthetic text data using:
- LLM-based generation (Anthropic Claude, OpenAI, etc.)
- Template-based generation
- Length control
- Style preservation
"""

import os
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TextSample:
    """Sample text with metadata"""
    text: str
    length: int
    word_count: int
    style_features: Dict[str, Any]


class LLMEngine:
    """
    LLM abstraction layer
    
    Supports multiple LLM providers:
    - Anthropic Claude
    - OpenAI GPT
    - HuggingFace models
    - Local models
    """
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        """
        Initialize LLM engine
        
        Args:
            provider: LLM provider name
            model: Model identifier
            api_key: API key (reads from env if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Get API key from environment if not provided
        if api_key is None:
            if self.provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
        
        self.api_key = api_key
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        try:
            if self.provider == "anthropic":
                try:
                    from anthropic import Anthropic
                    if not self.api_key:
                        logger.warning("No Anthropic API key found, LLM generation will fail")
                        return None
                    return Anthropic(api_key=self.api_key)
                except ImportError:
                    logger.error("anthropic package not installed. Run: pip install anthropic")
                    return None
            
            elif self.provider == "openai":
                try:
                    from openai import OpenAI
                    if not self.api_key:
                        logger.warning("No OpenAI API key found, LLM generation will fail")
                        return None
                    return OpenAI(api_key=self.api_key)
                except ImportError:
                    logger.error("openai package not installed. Run: pip install openai")
                    return None
            
            elif self.provider == "huggingface":
                logger.warning("HuggingFace provider not yet implemented")
                return None
            
            elif self.provider == "local":
                logger.info("Using local generation (placeholder)")
                return None
            
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Generate text using LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_retries: Number of retries on failure
            
        Returns:
            Generated text or None on failure
        """
        if self.client is None:
            return self._fallback_generation(prompt)
        
        for attempt in range(max_retries):
            try:
                if self.provider == "anthropic":
                    return self._generate_anthropic(prompt, system_prompt)
                elif self.provider == "openai":
                    return self._generate_openai(prompt, system_prompt)
                else:
                    return self._fallback_generation(prompt)
                    
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("All generation attempts failed")
                    return self._fallback_generation(prompt)
        
        return None
    
    def _generate_anthropic(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Anthropic Claude"""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
    
    def _generate_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using OpenAI"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def _fallback_generation(self, prompt: str) -> str:
        """Simple fallback generation when LLM is unavailable"""
        # Extract key words from prompt
        words = re.findall(r'\w+', prompt.lower())
        
        # Generate simple text based on prompt keywords
        templates = [
            f"This is a sample text about {' and '.join(words[:3])}.",
            f"Generated content regarding {words[0] if words else 'topic'}.",
            f"Synthetic text example for data generation purposes.",
        ]
        
        return np.random.choice(templates)
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        delay: float = 0.1
    ) -> List[str]:
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            delay: Delay between requests to avoid rate limits
            
        Returns:
            List of generated texts
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt, system_prompt)
            results.append(result or "")
            
            # Rate limiting
            if i < len(prompts) - 1 and delay > 0:
                time.sleep(delay)
        
        return results


class TemplateEngine:
    """
    Template-based text generation
    
    Uses templates with placeholders for faster, deterministic generation
    """
    
    def __init__(self, templates_path: Optional[Path] = None):
        """
        Initialize template engine
        
        Args:
            templates_path: Path to templates JSON file
        """
        self.templates = self._load_templates(templates_path)
    
    def _load_templates(self, templates_path: Optional[Path]) -> Dict[str, List[str]]:
        """Load templates from file or use defaults"""
        if templates_path and templates_path.exists():
            with open(templates_path, 'r') as f:
                return json.load(f)
        
        # Default templates
        return {
            'product_review': [
                "This {adjective} product {verb} my expectations. {detail}",
                "I {feeling} this purchase. The quality is {quality}.",
                "{adjective} experience overall. Would {recommendation}.",
            ],
            'customer_feedback': [
                "The service was {quality}. {detail}",
                "{feeling} about the {aspect}. {recommendation}",
                "My experience was {adjective}. {detail}",
            ],
            'description': [
                "A {adjective} {noun} that {verb} {adverb}.",
                "This {noun} is {adjective} and {adjective2}.",
                "{adjective} {noun} with {feature}.",
            ],
            'comment': [
                "{adjective} content! {detail}",
                "This is {feeling}. {reason}",
                "{adjective} post. {reaction}",
            ]
        }
    
    def generate(
        self,
        category: str = 'description',
        context: Optional[Dict[str, Any]] = None,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate text from templates
        
        Args:
            category: Template category
            context: Context variables for placeholders
            num_samples: Number of samples to generate
            
        Returns:
            List of generated texts
        """
        if category not in self.templates:
            category = list(self.templates.keys())[0]
        
        templates = self.templates[category]
        results = []
        
        for _ in range(num_samples):
            template = np.random.choice(templates)
            text = self._fill_template(template, context)
            results.append(text)
        
        return results
    
    def _fill_template(self, template: str, context: Optional[Dict[str, Any]]) -> str:
        """Fill template placeholders"""
        # Default fill values
        defaults = {
            'adjective': ['great', 'excellent', 'good', 'amazing', 'wonderful', 'poor', 'bad'],
            'adjective2': ['useful', 'practical', 'innovative', 'reliable'],
            'verb': ['exceeded', 'met', 'disappointed'],
            'feeling': ['love', 'like', 'appreciate', 'regret'],
            'quality': ['excellent', 'good', 'acceptable', 'poor'],
            'detail': ['Highly recommended.', 'Worth the price.', 'Could be better.'],
            'recommendation': ['recommend it', 'buy again', 'pass on this'],
            'aspect': ['product', 'service', 'delivery', 'quality'],
            'noun': ['product', 'item', 'service', 'solution'],
            'adverb': ['perfectly', 'well', 'adequately'],
            'feature': ['many features', 'great design', 'solid build'],
            'reason': ['Just what I needed.', 'Exactly as described.'],
            'reaction': ['Thanks for sharing!', 'Interesting perspective.'],
        }
        
        # Merge with context
        if context:
            defaults.update(context)
        
        # Find all placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Fill each placeholder
        text = template
        for placeholder in placeholders:
            if placeholder in defaults:
                value = defaults[placeholder]
                if isinstance(value, list):
                    value = np.random.choice(value)
                text = text.replace(f'{{{placeholder}}}', str(value))
        
        return text


class TextGenerator:
    """
    Main text data generator
    
    Combines LLM and template-based generation with length control
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize text generator
        
        Args:
            config: Configuration object with text settings
        """
        self.config = config
        
        # Initialize engines
        if config and hasattr(config, 'text'):
            self.llm_engine = LLMEngine(
                provider=config.text.llm_provider,
                model=config.text.model,
                max_tokens=config.text.max_tokens,
                temperature=config.text.temperature
            )
            self.use_templates = config.text.use_templates
            self.preserve_length = config.text.preserve_length
            self.length_variance = config.text.length_variance
        else:
            self.llm_engine = LLMEngine()
            self.use_templates = False
            self.preserve_length = True
            self.length_variance = 0.2
        
        self.template_engine = TemplateEngine()
    
    def generate(
        self,
        reference_data: pd.Series,
        num_rows: int,
        column_name: str = "text",
        seed: Optional[int] = None
    ) -> pd.Series:
        """
        Generate text data
        
        Args:
            reference_data: Reference text data
            num_rows: Number of texts to generate
            column_name: Name of the column
            seed: Random seed
            
        Returns:
            Series of generated texts
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Generating {num_rows} text samples for column '{column_name}'")
        
        # Analyze reference data
        samples = self._analyze_samples(reference_data)
        
        # Choose generation method
        if self.use_templates:
            generated = self._generate_with_templates(samples, num_rows)
        else:
            generated = self._generate_with_llm(samples, num_rows, column_name)
        
        # Adjust lengths if needed
        if self.preserve_length:
            generated = self._adjust_lengths(generated, samples)
        
        return pd.Series(generated, name=column_name)
    
    def _analyze_samples(self, reference: pd.Series) -> List[TextSample]:
        """Analyze reference text samples"""
        samples = []
        
        clean_data = reference.dropna().astype(str)
        
        for text in clean_data.head(20):  # Sample first 20
            sample = TextSample(
                text=text,
                length=len(text),
                word_count=len(text.split()),
                style_features=self._extract_style_features(text)
            )
            samples.append(sample)
        
        return samples
    
    def _extract_style_features(self, text: str) -> Dict[str, Any]:
        """Extract style features from text"""
        return {
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'has_punctuation': any(c in text for c in '.,!?;:'),
            'has_numbers': any(c.isdigit() for c in text),
            'is_uppercase': text.isupper(),
            'is_lowercase': text.islower(),
            'capitalized_words': sum(1 for word in text.split() if word and word[0].isupper()),
        }
    
    def _generate_with_templates(
        self,
        samples: List[TextSample],
        num_rows: int
    ) -> List[str]:
        """Generate using templates"""
        logger.info("Using template-based generation")
        
        # Determine template category from samples
        category = 'description'  # Default
        
        return self.template_engine.generate(
            category=category,
            num_samples=num_rows
        )
    
    def _generate_with_llm(
        self,
        samples: List[TextSample],
        num_rows: int,
        column_name: str
    ) -> List[str]:
        """Generate using LLM"""
        logger.info("Using LLM-based generation")
        
        # Build prompt from samples
        system_prompt = self._build_system_prompt(samples, column_name)
        
        # Generate in batches
        batch_size = 10
        all_generated = []
        
        for i in range(0, num_rows, batch_size):
            batch_size_actual = min(batch_size, num_rows - i)
            
            prompt = f"Generate {batch_size_actual} diverse examples of {column_name}. Each on a new line."
            
            response = self.llm_engine.generate(prompt, system_prompt)
            
            if response:
                # Split by newlines and clean
                texts = [line.strip() for line in response.split('\n') if line.strip()]
                
                # Remove numbering if present
                texts = [re.sub(r'^\d+[\.\)]\s*', '', text) for text in texts]
                
                all_generated.extend(texts[:batch_size_actual])
            else:
                # Fallback to templates
                logger.warning(f"LLM generation failed for batch {i}, using templates")
                all_generated.extend(
                    self.template_engine.generate(num_samples=batch_size_actual)
                )
        
        # Ensure we have exact number needed
        if len(all_generated) < num_rows:
            # Pad with templates
            remaining = num_rows - len(all_generated)
            all_generated.extend(self.template_engine.generate(num_samples=remaining))
        
        return all_generated[:num_rows]
    
    def _build_system_prompt(self, samples: List[TextSample], column_name: str) -> str:
        """Build system prompt from sample analysis"""
        if not samples:
            return f"Generate diverse, realistic examples of {column_name}."
        
        avg_length = np.mean([s.length for s in samples])
        avg_words = np.mean([s.word_count for s in samples])
        
        # Get sample texts
        example_texts = [s.text for s in samples[:3]]
        
        prompt = f"""Generate realistic synthetic data for a column named '{column_name}'.

Style guidelines based on reference data:
- Average length: {avg_length:.0f} characters
- Average words: {avg_words:.0f} words
- Similar style and tone to these examples:

Examples:
{chr(10).join(f'  - {text}' for text in example_texts)}

Generate diverse but stylistically consistent examples. Be concise and natural."""
        
        return prompt
    
    def _adjust_lengths(
        self,
        generated: List[str],
        samples: List[TextSample]
    ) -> List[str]:
        """Adjust text lengths to match reference distribution"""
        if not samples:
            return generated
        
        target_lengths = [s.length for s in samples]
        avg_target = np.mean(target_lengths)
        std_target = np.std(target_lengths)
        
        adjusted = []
        
        for text in generated:
            current_length = len(text)
            
            # Calculate target length with variance
            variance = np.random.uniform(-self.length_variance, self.length_variance)
            target_length = int(avg_target * (1 + variance))
            
            # Adjust if needed
            if current_length < target_length * 0.8:
                # Too short - repeat or pad
                text = self._lengthen_text(text, target_length)
            elif current_length > target_length * 1.2:
                # Too long - truncate
                text = self._shorten_text(text, target_length)
            
            adjusted.append(text)
        
        return adjusted
    
    def _lengthen_text(self, text: str, target_length: int) -> str:
        """Lengthen text to target length"""
        while len(text) < target_length:
            # Add filler phrases
            fillers = [
                " Additionally,", " Furthermore,", " Moreover,",
                " In addition,", " Also,", " Notably,"
            ]
            text += np.random.choice(fillers) + " " + text.split('.')[0] + "."
        
        return text[:target_length]
    
    def _shorten_text(self, text: str, target_length: int) -> str:
        """Shorten text to target length"""
        if len(text) <= target_length:
            return text
        
        # Try to cut at sentence boundary
        sentences = text.split('.')
        result = ""
        
        for sentence in sentences:
            if len(result) + len(sentence) < target_length:
                result += sentence + "."
            else:
                break
        
        # If still too long, hard truncate
        if len(result) > target_length:
            result = text[:target_length].rsplit(' ', 1)[0] + "..."
        
        return result


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample reference data
    reference_texts = pd.Series([
        "Great product, highly recommended!",
        "Excellent quality and fast shipping.",
        "Not what I expected, disappointed.",
        "Amazing value for the price.",
        "Good but could be better.",
    ])
    
    # Generate with templates (no API key needed)
    from dataclasses import dataclass, field
    
    @dataclass
    class MockTextConfig:
        llm_provider: str = "anthropic"
        model: str = "claude-sonnet-4-20250514"
        max_tokens: int = 100
        temperature: float = 0.7
        use_templates: bool = True
        preserve_length: bool = True
        length_variance: float = 0.2
    
    @dataclass
    class MockConfig:
        text: MockTextConfig = field(default_factory=MockTextConfig)
    
    config = MockConfig()
    generator = TextGenerator(config)
    
    synthetic = generator.generate(
        reference_data=reference_texts,
        num_rows=10,
        column_name="review"
    )
    
    print("\nGenerated reviews:")
    print("="*60)
    for i, text in enumerate(synthetic, 1):
        print(f"{i}. {text}")
