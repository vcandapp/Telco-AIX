# System Prompts Configuration

## Overview
The SME Web UI now supports loading system prompts from an external JSON file (`system_prompts.json`), making it easy to customize and manage different AI personas without modifying the code.

## File Structure
The system prompts are stored in `system_prompts.json` with the following structure:

```json
{
    "Prompt Name": "Prompt content with full description...",
    "Another Prompt": "Another prompt content..."
}
```

## Available System Prompts

1. **Default Assistant** - General-purpose AI assistant with systematic approach
2. **Technical Expert** - Senior technical expert for architecture and engineering
3. **Code Assistant** - Expert programmer for production-ready code
4. **Data Analyst** - Data analysis and business insights expert
5. **Creative Writer** - Strategic content creation expert
6. **Network Expert** - Network architecture and engineering specialist
7. **Telco Expert** - Telecommunications solutions architect
8. **Custom** - Placeholder for user-defined prompts

## Features

### 1. External File Loading
- System prompts are loaded from `system_prompts.json` at startup
- Falls back to default prompts if file is not found or has errors

### 2. Dynamic Reload
- Click "🔄 Reload Prompts" button to reload prompts from file without restarting
- Useful when editing the JSON file directly

### 3. Save Custom Prompts
- Use the "💾 Save Custom Prompt" accordion in the UI
- Enter a name and content for your custom prompt
- Click "💾 Save Prompt" to add it to the system_prompts.json file
- The dropdown will automatically update with the new prompt

### 4. Edit Existing Prompts
- Edit the `system_prompts.json` file directly
- Click "🔄 Reload Prompts" to apply changes

## Usage Examples

### Adding a New Prompt via UI
1. Expand "💾 Save Custom Prompt" section
2. Enter prompt name: "Security Expert"
3. Enter prompt content: "You are a cybersecurity expert..."
4. Click "💾 Save Prompt"
5. Select "Security Expert" from the dropdown

### Editing Prompts Directly
1. Open `system_prompts.json` in a text editor
2. Modify existing prompts or add new ones
3. Save the file
4. Click "🔄 Reload Prompts" in the UI

### Using Custom Prompts
1. Select a prompt from the dropdown OR
2. Enter a custom prompt in the "Custom System Prompt" text area
3. Custom prompt text always overrides the selected template

## Best Practices

1. **Backup**: Keep a backup of your customized `system_prompts.json`
2. **Formatting**: Use proper JSON formatting with escaped characters
3. **Testing**: Test new prompts before saving to ensure they work as expected
4. **Organization**: Use clear, descriptive names for your prompts
5. **Version Control**: Track changes to system_prompts.json in git

## Troubleshooting

- **Prompts not loading**: Check JSON syntax in system_prompts.json
- **Changes not appearing**: Click "🔄 Reload Prompts" after editing
- **Error messages**: Check the console output for detailed error information
- **Default prompts appear**: Indicates an issue with the JSON file - check syntax

## JSON Formatting Tips

- Use `\n` for line breaks within prompt content
- Escape quotes with `\"`
- Use online JSON validators to check syntax
- Keep prompts concise but comprehensive