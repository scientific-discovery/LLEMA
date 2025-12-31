from agent.llm_interface import call_llm
import os
import random
import re
from agent.utils import try_parse_json_response, extract_json_from_llm

# Load rules text from generation_rules.md
RULES_PATH = os.path.join(os.path.dirname(__file__), 'generation_rules.md')
with open(RULES_PATH, 'r') as f:
    RULES_TEXT = f.read().split('---')[0]  # Use the rules section only

RETRY_EMPTY_LIMIT = 3

def extract_and_randomize_rules(rules_text: str, num_rules: int = 6) -> str:
    """
    Extract individual rules from the rules text, randomly select num_rules,
    and reorder them to counter positional bias.
    """
    # Extract individual rules using regex
    rule_pattern = r'Rule \d+:.*?(?=Rule \d+:|$)'
    rules = re.findall(rule_pattern, rules_text, re.DOTALL)
    
    # Clean up the rules (remove extra whitespace and newlines)
    cleaned_rules = []
    for rule in rules:
        cleaned_rule = rule.strip()
        if cleaned_rule:
            cleaned_rules.append(cleaned_rule)
    
    # Randomly select num_rules rules
    if len(cleaned_rules) < num_rules:
        selected_rules = cleaned_rules
    else:
        selected_rules = random.sample(cleaned_rules, num_rules)
    
    # Shuffle the selected rules to counter positional bias
    random.shuffle(selected_rules)
    
    # Join the rules with proper formatting
    randomized_rules_text = '\n\n'.join(selected_rules)
    
    return randomized_rules_text

def evolve_candidates(evolution_memory: dict, constraints: str, n=5, buffer=None, island_id=None, iteration=0, task_name=None) -> list:
    # Convert constraints to natural language if task_name is provided
    if task_name:
        try:
            from agent.main import constraints_to_natural_language
            constraints = constraints_to_natural_language(task_name)
        except ImportError:
            # Fallback to original constraints if import fails
            pass
    
    # Use unique examples from island-specific memory if available
    if buffer is not None and island_id is not None:
        unique_examples = buffer.get_unique_examples_for_evolution(island_id, max_examples=20)
        if unique_examples['success'] or unique_examples['failure']:
            print(f"[Agent] Using {len(unique_examples['success'])} unique success and {len(unique_examples['failure'])} unique failure examples from island {island_id}")
            evolution_memory = unique_examples
    
    # Randomly select and shuffle 6 rules to counter positional bias
    randomized_rules = extract_and_randomize_rules(RULES_TEXT, num_rules=6)
    
    # Debug: Print which rules were selected (first line of each rule)
    rule_lines = randomized_rules.split('\n\n')
    selected_rule_numbers = []
    for rule in rule_lines:
        if rule.strip().startswith('Rule '):
            rule_num = rule.strip().split(':')[0]
            selected_rule_numbers.append(rule_num)
    print(f"[Agent] Selected rules for evolution: {', '.join(selected_rule_numbers)}")
    
    # Clean up examples to show molecular formulas with property values
    success_descriptions = []
    for success in evolution_memory['success']:
        formula = success.get('formula', success.get('compound', 'Unknown'))
        if formula != 'Unknown':
            # Extract property values from predictions
            property_values = success.get('property_values', {})
            if property_values:
                # Create natural language description of properties
                prop_descriptions = []
                for prop, value in property_values.items():
                    if value is not None:
                        # Map property names to human-readable names
                        prop_mapping = {
                            'band_gap': 'band gap',
                            'formation_energy': 'formation energy',
                            'energy_above_hull': 'energy above hull',
                            'dielectric_constant': 'dielectric constant',
                            'bulk_modulus': 'bulk modulus',
                            'shear_modulus': 'shear modulus',
                            'density': 'density'
                        }
                        readable_prop = prop_mapping.get(prop, prop)
                        prop_descriptions.append(f"{readable_prop} = {value:.3f}")
                
                if prop_descriptions:
                    success_descriptions.append(f"- {formula} has {', '.join(prop_descriptions)}")
                else:
                    success_descriptions.append(f"- {formula}")
            else:
                success_descriptions.append(f"- {formula}")
    
    failure_descriptions = []
    for failure in evolution_memory['failure']:
        formula = failure.get('formula', failure.get('compound', 'Unknown'))
        if formula != 'Unknown':
            # Extract property values from predictions
            property_values = failure.get('property_values', {})
            if property_values:
                # Create natural language description of properties
                prop_descriptions = []
                for prop, value in property_values.items():
                    if value is not None:
                        # Map property names to human-readable names
                        prop_mapping = {
                            'band_gap': 'band gap',
                            'formation_energy': 'formation energy',
                            'energy_above_hull': 'energy above hull',
                            'dielectric_constant': 'dielectric constant',
                            'bulk_modulus': 'bulk modulus',
                            'shear_modulus': 'shear modulus',
                            'density': 'density'
                        }
                        readable_prop = prop_mapping.get(prop, prop)
                        prop_descriptions.append(f"{readable_prop} = {value:.3f}")
                
                if prop_descriptions:
                    failure_descriptions.append(f"- {formula} has {', '.join(prop_descriptions)}")
                else:
                    failure_descriptions.append(f"- {formula}")
            else:
                failure_descriptions.append(f"- {formula}")
    
    # Format as simple text lists
    cleaned_successes = '\n'.join(success_descriptions) if success_descriptions else "None"
    cleaned_failures = '\n'.join(failure_descriptions) if failure_descriptions else "None"
    
    # Random prompt switching (1/3 chance for complex prompt, 2/3 for standard)
    if iteration < 10:
        use_complex_prompt = False
    else:
        use_complex_prompt = random.randint(0, 3) == 0
    
    if use_complex_prompt:
        print("[Agent] Using COMPLEX prompt to encourage advanced rule usage")
        prompt = f"""
### ROLE ###
You are an expert materials discovery assistant with deep knowledge of crystal chemistry and materials design.

### TASK ###
Your task is to generate materials for the following task:
Task: {task_name or 'Materials Discovery'}

### GOAL ###
Using the design rules and prior outcomes, propose {n} new candidate materials that are likely to satisfy the given constraints while maximizing innovation and chemical plausibility for this specific task.

### INPUTS ###
- Here are a few design rules for generating new compounds:
{randomized_rules}

- The following are the design constraints for this task:
{constraints}

- Top successful candidates (where constraints were satisfied):
{cleaned_successes}

- Top failed candidates (where one or more constraints were not satisfied):
{cleaned_failures}

Given the following top successful and failed candidates and the provided evolutionary templates, generate {n} new candidate materials that are likely to satisfy the constraints.

IMPORTANT: 
    - You should PRIORITIZE using the MORE ADVANCED rules (Rules 4-14) when possible, but you can still use simpler rules (Rules 1-3) if they are the most appropriate for the given context. 
    - The advanced rules require deeper chemical understanding and often lead to more innovative materials.

### INSTRUCTIONS ###
- FAVOR Rules 4-14 which involve: functional groups, crystal prototypes, coordination geometry, 
  oxidation/reduction variants, surface functionalization, template-guided design, property conditioning, 
  retrosynthesis, functional analogs, tolerance factors, and periodicity
- Use Rules 1-3 (simple substitutions) only when they are clearly the best choice
- Apply sophisticated chemical reasoning and consider structural stability, coordination environments, and electronic properties
- Balance innovation with chemical validity

### OUTPUT FORMAT ###
- Select the most appropriate rule(s) or a combination of rules from the 6 rules provided above.
- For each new compound, return a JSON array only (no markdown, no prose) as follows:
  {{
    "compound": string, # proposed chemical formula
    "rules_used": string, # rule number(s) and/or text
    "justification": string # detailed chemical reasoning in 50 to 100 words
  }}

Return a JSON list of new candidates, each with 'compound', 'rules_used', and 'justification'.
"""
    else:
        print("[Agent] Using STANDARD prompt")
        prompt = f"""
### ROLE ###
You are an expert materials discovery assistant with deep knowledge of crystal chemistry and materials design.

### TASK ###
Your task is to generate materials for the following task:
Task: {task_name or 'Materials Discovery'}

### GOAL ###
Using the design rules and prior outcomes, propose {n} new candidate materials that are likely to satisfy the given constraints while maximizing innovation and chemical plausibility for this specific task.

### INPUTS ###
- Here are a few design rules for generating new compounds:
{randomized_rules}

- The following are the design constraints for this task:
{constraints}

- Top successful candidates (where constraints were satisfied):
{cleaned_successes}

- Top failed candidates (where constraints were not satisfied):
{cleaned_failures}

Given the following top successful and failed candidates and the provided evolutionary templates, generate {n} new candidate materials that are likely to satisfy the constraints.

### OUTPUT FORMAT ###
- Select the most appropriate rule(s) or a combination of rules from the 6 rules provided above.
- For each new compound, return a JSON array only (no markdown, no prose) as follows:
  {{
    "compound": string, # proposed chemical formula
    "rules_used": string, # rule number(s) and/or text
    "justification": string # detailed chemical reasoning in 50 to 100 words
  }}

Return a JSON list of new candidates, each with 'compound', 'rules_used', and 'justification'.
""" 
    print(f"[Agent] Using prompt: {prompt}")
    for attempt in range(RETRY_EMPTY_LIMIT):
        response = call_llm(prompt)
        # Try to extract and parse JSON from the response
        cleaned = extract_json_from_llm(response)
        parsed = try_parse_json_response(cleaned, f"Evolution LLM attempt {attempt+1}")
        if parsed:
            return parsed
        else:
            # Log the raw response for debugging
            with open('evolution_llm_parse_errors.log', 'a') as f:
                f.write(f'Attempt {attempt+1}:\nPrompt:\n{prompt}\nResponse:\n{response}\n---\n')
    # If all attempts fail, return empty list
    return []

def filter_duplicates(candidates: list, buffer=None, island_id=None) -> list:
    """Filter out duplicate candidates using island memory"""
    if buffer is None or island_id is None:
        return candidates
    
    unique_candidates = []
    for candidate in candidates:
        if not buffer.is_duplicate_candidate(island_id, candidate):
            unique_candidates.append(candidate)
        else:
            formula = candidate.get('formula', candidate.get('compound', 'Unknown'))
            print(f"[Agent] Filtered duplicate candidate: {formula}")
    
    print(f"[Agent] Filtered {len(candidates) - len(unique_candidates)} duplicates, {len(unique_candidates)} unique candidates remain")
    return unique_candidates