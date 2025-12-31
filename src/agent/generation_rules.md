# Materials Discovery Generation Rules

Your task is to generate a new, chemically valid, stoichiometric compound based on a known compound and a specific design rule.

Rules describe chemically meaningful transformations. The output compound must:
- Follow the selected rule exactly
- Be stoichiometric and ordered (no alloys or partial occupancies)
- Be compatible, meaning it could correspond to a valid 3D crystal structure
- Include a short justification

## List of Allowed Generation Rules

Rule 1: Same-group elemental substitution: Replace each element with another from the same periodic group. Example: A₂B₃ → C₂D₃, where C and D are in the same groups as A and B.Example: A₂B₃ → C₂D₃, where C and D are in the same groups as A and B.

Rule 2: Stoichiometry-preserving substitution: Keep the formula ratios but use chemically similar elements. Example: A₂B₃C₄ → D₂E₃F₄, where D, E, F are similar to A, B, C.

Rule 3: Oxidation state substitution: Replace each element with one having the same oxidation state. Example: A²⁺B⁻ → C²⁺D⁻.

Rule 4: Functional group substitution: Swap one functional group with another of similar chemical behavior. Example: R–X → R–Y, where X and Y are functionally analogous.

Rule 5: Crystal prototype substitution: Maintain the structural prototype (e.g., ABX₃) and replace elements. Example: ABX₃ → CDY₃.

Rule 6: Coordination geometry mutation: Change the ligand geometry around the central atom. Example: A(L)₄ → A(L)₆.

Rule 7: Oxidation/reduction variant: Adjust stoichiometry for a different redox configuration. Example: A₂B₃ → A₃B₄.

Rule 8: Surface functionalization: Add functional groups to a known material's surface. Example: ABC → ABC–X.

Rule 9: Template-guided combinatorics: Fill in a known formula structure with compatible elements. Example: ABX₃ → C–D–E₃.

Rule 10: Inverse property conditioning: Generate a material likely to exhibit a specified property. Example: Target: high hardness → A₂B.

Rule 11: Retrosynthesis-based forward design: Suggest a possible product from precursors. Example: A + B → C.",

Rule 12: Functional analog discovery: Replace a compound with another that serves the same function. Example: A₂B₃ (insulator) → C₄D₆ (also insulator).

Rule 13: Tolerance-factor guided substitution: Replace atoms while preserving structural stability rules. Example: ABX₃ → A′BX₃ (A′ has similar ionic radius as A).

Rule 14: Periodicity-preserving analog search: Replace atoms while maintaining periodic trends. Example: A₂B₃ → C₂D₃, where C and D are periodic analogs of A and B.

---

## LLM Prompt Template

Given the input below, generate:
1. A new compound (formula only)
2. A short justification for why the rule was followed correctly

Input compound: {KNOWN_COMPOUND}  
Selected rule: {RULE_NUMBER OR RULE_TEXT}

Output:
Compound: <your output>  
Justification: <your reasoning> 