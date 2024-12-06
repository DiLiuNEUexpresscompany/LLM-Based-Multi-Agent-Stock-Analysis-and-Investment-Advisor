from tools.format_tool import FormatConvertTool
from agents.format_convert_agent import FormatConverter
from agents.tool_registry import ToolRegistry
import json

# Create a tool registry
registry = ToolRegistry()

# Register the LLM text converter tool
registry.register(FormatConvertTool())

# Create the format finder agent
format_agent = FormatConverter(registry)

# Dummy example: Convert a recipe description to JSON
# Extended example: Convert a recipe description to JSON
recipe_text = """
Hearty Vegetable Soup with Lentils
This comforting soup is perfect for chilly days and packed with wholesome goodness.
Ingredients:
- 2 tablespoons olive oil
- 1 onion, diced
- 2 carrots, sliced
- 2 celery stalks, chopped
- 3 cloves garlic, minced
- 1 cup dried lentils
- 1 can (14 oz) diced tomatoes
- 4 cups vegetable broth
- 2 teaspoons ground cumin
- 1 teaspoon smoked paprika
- 1/2 teaspoon black pepper
- 1 teaspoon salt (adjust to taste)
- 1 cup spinach leaves
Instructions:
1. Heat olive oil in a large pot over medium heat.
2. Add onion, carrots, and celery; sauté until softened, about 5 minutes.
3. Stir in garlic, cumin, smoked paprika, salt, and pepper; cook for 1 minute.
4. Add lentils, diced tomatoes, and vegetable broth; bring to a boil.
5. Reduce heat to low, cover, and simmer for 30 minutes or until lentils are tender.
6. Stir in spinach leaves and cook for another 2 minutes.
7. Serve warm, garnished with fresh herbs if desired.
This recipe yields about 6 servings.
"""

# Convert to JSON with specific instructions
response = format_agent.run(
    text=recipe_text,
    output_format='json',
    conversion_instructions='Extract recipe details, ingredients, and step-by-step cooking instructions'
)

print("=" * 100)

with open('data/output.json', 'w') as f:
    json.dump(response, f, indent=2)
print(response)