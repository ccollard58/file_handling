# Real-Time Logging Tab Feature

## Overview
The Document Organizer GUI now includes a real-time logging tab that displays application logs as they occur. This feature provides immediate visibility into the application's operations, making it easier to monitor processing and troubleshoot issues.

## Features

### Real-Time Log Display
- **Live Updates**: Log messages appear instantly in the logging tab as they are generated
- **Color Coding**: Different log levels are color-coded for easy identification:
  - DEBUG: Gray (#888888)
  - INFO: White (#ffffff) 
  - WARNING: Orange (#ffaa00)
  - ERROR: Red (#ff6666)
  - CRITICAL: Bright Red (#ff0000)

### Global Log Level Setting
- **Settings Integration**: Configure the minimum log level through the expandable Settings panel
- **Persistent Configuration**: Log level setting is saved and restored between application sessions
- **Real-Time Application**: Changes apply immediately to both the logging display and the underlying logging system
- **Available Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Log Level Filtering
- **Dynamic Filtering**: Use the dropdown in the logging tab to filter logs by minimum level
- **Independent Control**: Logging tab filter works independently of the global log level setting
- **Real-Time**: Filter changes apply immediately to both new and existing messages

### Log Management
- **Clear Logs**: Remove all displayed log messages with the "Clear Logs" button
- **Save Logs**: Export current log messages to a text file with timestamp
- **Memory Management**: Automatically limits to last 1000 messages to prevent memory issues

### User Interface
- **Tabbed Interface**: Separate tab for logging doesn't interfere with main workflow
- **Dark Theme**: Easier on the eyes with a dark background and monospace font
- **Auto-Scroll**: Automatically scrolls to show the latest log messages

## Usage

### Setting the Global Log Level
1. Launch the Document Organizer application
2. In the main tab, click the "Settings..." button to expand the settings panel
3. Find the "Log Level" dropdown in the settings grid
4. Select your desired minimum log level (DEBUG shows all messages, CRITICAL shows only critical errors)
5. Click "Apply Settings" to save and apply the new log level
6. The setting is automatically saved and will be restored when you restart the application

### Accessing the Logging Tab
1. Launch the Document Organizer application
2. Click on the "Logging" tab at the top of the window
3. Real-time logs will begin appearing immediately based on your configured log level

### Filtering Logs in the Logging Tab
1. Use the "Log Level" dropdown in the logging tab header
2. Select the minimum level you want to see (e.g., selecting "WARNING" shows WARNING, ERROR, and CRITICAL messages)
3. The display updates immediately
4. This filter is independent of the global log level setting in the Settings panel

### Saving Logs
1. Click the "Save Logs..." button in the logging tab
2. Choose a location and filename for the log file
3. All current log messages will be saved in chronological order

## Configuration Details

### Settings Panel Integration
The log level setting is integrated into the main settings panel alongside:
- **Text Model**: LLM model selection for document analysis
- **Vision Model**: Vision model selection for image processing
- **Temperature**: LLM temperature setting for response variability
- **Log Level**: Minimum log level for application logging

### Configuration Persistence
The log level setting is automatically saved to the configuration file (`~/.document_organizer_config.json`) and includes:
- **Automatic Save**: Settings are saved when you click "Apply Settings"
- **Startup Restoration**: Log level is restored when the application starts
- **Cross-Session Persistence**: Your log level preference is maintained between application sessions

## Benefits

### For Users
- **Transparency**: See exactly what the application is doing during file processing
- **Customizable Verbosity**: Control how much detail you want to see in the logs
- **Progress Monitoring**: Monitor analysis and file processing operations in real-time
- **Error Detection**: Immediately spot warnings or errors during operation

### For Troubleshooting
- **Detailed Information**: Use DEBUG level for maximum detail when troubleshooting
- **Focused View**: Use higher levels (WARNING/ERROR) to focus on issues only
- **Export Logs**: Save logs for technical support or issue reporting
- **Historical View**: Review what happened during a processing session

## Log Level Descriptions

### DEBUG
- **Most Verbose**: Shows all log messages including detailed internal operations
- **Use Case**: Troubleshooting, development, detailed analysis of application behavior
- **Performance**: May generate many messages, can slow down processing slightly

### INFO (Default)
- **Balanced**: Shows informational messages about major operations and progress
- **Use Case**: Normal operation monitoring, tracking document processing progress
- **Performance**: Moderate message volume, minimal performance impact

### WARNING
- **Issues Focus**: Shows warnings about potential problems and actual errors
- **Use Case**: Monitoring for issues while filtering out routine information
- **Performance**: Low message volume, no performance impact

### ERROR
- **Error Focus**: Shows only error messages and critical issues
- **Use Case**: Problem diagnosis, error monitoring
- **Performance**: Very low message volume

### CRITICAL
- **Critical Only**: Shows only critical system errors that may cause application failure
- **Use Case**: Minimal logging, production environments
- **Performance**: Minimal message volume

## Technical Implementation

### Settings Integration
- Integrated into the existing settings framework with automatic save/restore
- Connected to both the logging display and the underlying Python logging system
- Thread-safe updates to logging configuration

### Dual Control System
- **Global Setting**: Controls what messages are generated and captured by the logging system
- **Display Filter**: Controls what messages are shown in the logging tab interface
- Independent operation allows fine-grained control over logging verbosity

### Configuration Format
```json
{
  "log_level": "INFO",
  "output_folder": "E:\\scanned documents",
  "source_folder": "E:\\Documents",
  "llm_settings": {
    "model": "gemma3:latest",
    "vision_model": "llava:latest", 
    "temperature": 0.6
  }
}
```

## Log Message Examples

```
[14:30:15] INFO     | Starting Document Organizer application
[14:30:16] INFO     | LLM initialized successfully with model gemma3:latest
[14:30:16] DEBUG    | Starting new HTTP connection (1): localhost:11434
[14:30:17] INFO     | Log level updated to: DEBUG
[14:30:20] INFO     | Analyzing file: document.pdf
[14:30:22] WARNING  | OCR confidence low for page 2
[14:30:25] INFO     | File processed successfully: new_filename.pdf
[14:30:25] ERROR    | Failed to move file: permission denied
```

This enhanced logging system with configurable log levels provides both transparency and control, making it easy to monitor application behavior at the appropriate level of detail for any situation.
