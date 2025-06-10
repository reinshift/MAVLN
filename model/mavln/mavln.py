import torch
import torch.nn as nn
import os
import logging
from typing import List, Dict,  Union
from transformers import AutoProcessor, AutoModelForVision2Seq
from Mantis.mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
from Mantis.mantis.models.mllava import chat_mllava
from transformers.generation import GenerationConfig
from PIL import Image
import yaml

logger = logging.getLogger(__name__)

class MAVLN(nn.Module):
    """
    Multi-Agent Vision-Language Navigation model based on Mantis.
    """
    def __init__(self, config):
        super(MAVLN, self).__init__()
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.num_agents = config.data.num_agents
        self.action_map = config.model.action_map

        model_path = config.model.mantis_path
        if not os.path.exists(model_path):
            logger.error(f"Mantis model path {model_path} not found")

        if config.model.maitis_type == "Idefics2":
            self.processor = AutoProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Idefics2")
            if hasattr(self.processor, "tokenizer"):
                tokenizer = self.processor.tokenizer
            else:
                tokenizer = self.processor
                
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
            self.model.generation_config.max_new_tokens = 512
            self.model.generation_config.temperature = 0.7
            self.model.generation_config.top_p = 0.9            
        elif config.model.maitis_type == "default":
            self.processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
            attn_implementation = None
            self.model = LlavaForConditionalGeneration.from_pretrained(config.model.mantis_path,
                                                       device_map="auto",
                                                       torch_dtype=torch.bfloat16,
                                                       resume_download=True,
                                                       force_download=True,
                                                       attn_implementation=attn_implementation)

        logger.info("model initialized.")

    def build_prompt(self, agent_images: List[Image.Image], instructions: List[str] = None) -> str:
        """
        Build a prompt for multi-agent navigation.

        Args:
            agent_images: List of images for each agent [agent1_image, agent2_image, ...]
            instructions: List of instructions for each agent.

        Returns:
            Formatted prompt string.
        """
        if len(agent_images) != self.num_agents:
            logger.warning(f"Expected {self.num_agents} agent images, but got {len(agent_images)}")

        # System prompt
        prompt = "You are commanding mutiple UAVs to navigate in an environment. "
        prompt += "Please based on their current views and instructions, determine the best action for each UAV.\n\n"

        # each agent's view and instruction inserted
        for i, agent_img_list in enumerate(agent_images):
            agent_name = f"Agent {i+1}"
            prompt += f"This is {agent_name}'s view: \n"

            # images will be inserted here by the processor

            # add instruction
            if instructions and i < len(instructions):
                prompt += f"\n{agent_name}'s instruction: {instructions[i]}\n\n"
            else:
                prompt += f"\n{agent_name} has no specific instruction.\n\n"

        prompt += "Actions available for each UAV:\n"
        for action_id, action_name in self.action_map.items():
            prompt += f"{action_id}: {action_name}\n"

        prompt += "\nPlease give the action number each UAV should take according their view and instruction:\n"
        prompt += "Format your response as 'Agent 1: [action_number], Agent 2: [action_number], ...' "
        prompt += "Explain your reasoning for each agent's action.\n"

        return prompt

    def forward(
            self,
            agent_images: List[List[Image.Image]],
            instructions: List[str] = None,
            return_logits: bool = False, # TODO: post training will use logits
    ) -> Union[Dict[str, torch.Tensor], str]:
        """
        Args:
            agent_images: List of lists of images for each agent [agent1_image, agent2_image, ...]
            instructions: List of instructions for each agent.
            return_logits: Whether to return the logits (True) or generated text (False).
        
        Returns:
            Either model outputs with logits or generated text.
        """
        # Flatten the list of images for processing
        all_images = []
        for agent_img_list in agent_images:
            all_images.extend(agent_img_list)

        # build prompt
        prompt_text = self.build_prompt(agent_images, instructions)

        if self.config.model.maitis_type == "Idefics2":
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in all_images],
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=all_images, return_tensors="pt")
            if 'pixel_attention_mask' in inputs:
                del inputs['pixel_attention_mask']
                
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if return_logits:
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True
                    )
                return {
                    "logits": outputs.logits,
                    "hidden_states": outputs.hidden_states,
                    "image_hidden_states": outputs.image_hidden_states if hasattr(outputs, "image_hidden_states") else None
                }
            
            return self.generate_response(inputs)
        elif self.config.model.maitis_type == "default":
            generation_kwargs = {
                "max_new_tokens": 1024,
                "num_beams": 1,
                "do_sample": False
            }
            response, _ = chat_mllava(prompt_text, all_images, self.model, self.processor, **generation_kwargs)
            return response
    
    def generate_response(self, inputs: Dict[str, torch.Tensor]) -> str:
        """
        Returns: Generated text action_id.
        """
        generation_kwargs = {
            "max_new_tokens": self.model.generation_config.max_new_tokens,
            "num_beams": 1,
            "do_sample": True,
            "temperature": self.model.generation_config.temperature,
            "top_p": self.model.generation_config.top_p,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        response = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]

        return response

    def parse_actions(self,response: str) -> Dict[int, int]:
        """
        Parse the generated response to extract action numbers for each agent.

        Args:
            response: Generated text response.
        
        Returns:
            Dictionary mapping agent index to action number.
        """
        actions = {}

        # look for patterns
        import re
        action_patterns = re.findall(r"Agent\s+(\d+)\s*:\s*\[?(\d+)\]?", response)

        for agent_str, action_str in action_patterns:
            try:
                agent_idx = int(agent_str) - 1 # Convert to 0-based index
                action_num = int(action_str)
                actions[agent_idx] = action_num
            except ValueError:
                logger.warning(f"Invalid action format: {agent_str}: {action_str}")
                continue
        
        # Check if we found actions for all agents
        for i in range(self.num_agents):
            if i not in actions:
                logger.warning(f"No action found for Agent {i+1} in response")
                actions[i] = 0 # default to stop

        return actions

    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Load model from config file.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to object with attribute access
        class Config:
            def __init__(self, d):
                self._dict_props = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        # For dictionaries with integer keys (like action_map)
                        if any(isinstance(key, int) for key in v.keys()):
                            self._dict_props[k] = v
                        else:
                            setattr(self, k, Config(v))
                    else:
                        setattr(self, k, v)
            
            def __getattr__(self, name):
                # Check if this is a stored dictionary property
                if hasattr(self, '_dict_props') and name in self._dict_props:
                    return self._dict_props[name]
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        config = Config(config_dict)
        return cls(config)