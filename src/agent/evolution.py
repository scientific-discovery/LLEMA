from agent.llm_interface import call_llm
# Unused imports removed - no longer using os, random, or re
from agent.utils import try_parse_json_response, extract_json_from_llm

# Rules removed - no longer using generation rules

RETRY_EMPTY_LIMIT = 3

# Rules extraction function removed - no longer using rules

def evolve_candidates(evolution_memory: dict, constraints: str, n=5, buffer=None, island_id=None, iteration = 0,task_name=None) -> list:
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
            print(f"[Agent] Using {len(unique_examples['success']) + len(unique_examples['failure'])} unique examples from island {island_id}")
            evolution_memory = unique_examples
    
    # Rules removed - no longer using generation rules
    
    # Get a single sample from the immediate previous generation
    all_previous_candidates = evolution_memory.get('success', []) + evolution_memory.get('failure', [])
    
    # Select only one sample from the previous generation
    if all_previous_candidates:
        # Take the most recent candidate (last in the list)
        sample_candidate = all_previous_candidates[-1]
        
        formula = sample_candidate.get('formula', sample_candidate.get('compound', 'Unknown'))
        score = sample_candidate.get('score', 'Unknown')
        
        if formula != 'Unknown':
            # Extract property values from the sample
            property_values = sample_candidate.get('property_values', {})
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
                            'density': 'density',
                            'electrical_conductivity': 'electrical conductivity',
                            'piezo_max_dij': 'piezoelectric constant',
                            'piezo_max_dielectric': 'piezoelectric dielectric constant'
                        }
                        readable_prop = prop_mapping.get(prop, prop)
                        prop_descriptions.append(f"{readable_prop} = {value:.3f}")
                
                if prop_descriptions:
                    cleaned_previous = f"- {formula} with {', '.join(prop_descriptions)}"
                else:
                    cleaned_previous = f"- {formula}"
            else:
                cleaned_previous = f"- {formula}"
        else:
            cleaned_previous = "None"
    else:
        cleaned_previous = "None"
    
    # Simplified prompt focusing on band gap modification
    print("[Agent] Using BAND-GAP-MODIFICATION prompt")
    prompt = f"""
### ROLE ###
You are an expert materials discovery assistant with deep knowledge of crystal chemistry and materials design.

### TASK ###
Your task is to generate materials for the following task:
{constraints}

### GOAL ###
Please propose a modification to the material that satisfies the given constraints.
You can choose one of the four following modifications:
1. exchange: exchange two elements in the material
2. substitute: substitute one element in the material with another
3. remove: remove an element from the material
4. add: add an element to the material

- A sample from the previous generation with its property values:
{cleaned_previous}

### INSTRUCTIONS ###
- Analyze the sample from the previous generation and its properties
- Propose {n} modifications based on this single sample
- Choose the most appropriate modification type (exchange, substitute, remove, or add) for each proposal
- Apply chemical reasoning and consider structural stability, coordination environments, and electronic properties
- Ensure the modifications are chemically valid and likely to achieve the target properties

### OUTPUT FORMAT ###
- For each new compound, return a JSON array only (no markdown, no prose) as follows:
  {{
    "compound": string, # proposed chemical formula after modification
    "modification_type": string, # one of: "exchange", "substitute", "remove", "add"
    "justification": string # detailed chemical reasoning in 50 to 100 words explaining the modification
  }}

Return a JSON list of new candidates, each with 'compound', 'modification_type', and 'justification'.
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
            with open('evolution_llm_parse_errors.log', 'a', encoding='utf-8') as f:
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