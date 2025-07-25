﻿# Event Management AI Agent

A powerful AI-powered event scheduling and calendar management system that processes natural language requests to create, store, and manage events. Built with LangChain, OpenAI GPT, and Python.

## Features

- **Natural Language Processing**: Schedule events using plain English
- **Dual Storage**: Events saved in both ICS calendar format and CSV backup
- **Smart Extraction**: AI-powered extraction of event details from text
- **Flexible Queries**: List upcoming events with various time filters
- **Interactive CLI**: Easy-to-use command-line interface
- **Robust Fallback**: Fallback parsing when AI extraction fails

## Installation

1. Clone the repository:
```bash
git clone https://github.com/avinash00134/event-management-agent.git
cd event-management-agent
```

2. Install required dependencies:
```bash
pip install langchain langchain-openai pydantic ics pathlib
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Command Line Interface

Run the interactive agent:
```bash
python event_agent.py
```

### Example Commands

The agent understands natural language requests like:

- `"Schedule a meeting with John tomorrow at 2pm"`
- `"Add dentist appointment on July 25 at 9:30 AM at Downtown Clinic"`
- `"Team standup every Monday at 10am in Conference Room A"`
- `"Show my upcoming events"`
- `"List events for next week"`
- `"What do I have today?"`

### Programmatic Usage

```python
from event_agent import EventManagementAgent

# Initialize the agent
agent = EventManagementAgent(openai_api_key="your-api-key")

# Process event requests
result = agent.process_event_request("Schedule lunch with Sarah tomorrow at 1pm")
print(result)

# List upcoming events
events = agent.process_event_request("Show my events for next week")
print(events)
```

## Architecture

### Core Components

1. **EventDetails**: Pydantic model for structured event data
2. **EventStorage**: Handles dual storage in ICS and CSV formats
3. **EventManagementAgent**: Main agent orchestrating the workflow
4. **Extraction Tools**: AI-powered natural language processing

### Storage Files

- `events.ics`: Standard ICS calendar file for calendar applications
- `events_backup.csv`: CSV backup with full event details
- `event_log.txt`: Activity log for debugging and tracking

### Available Tools

- `extract_and_add_event`: Extract and immediately add events
- `extract_event_details_only`: Preview extraction without adding
- `list_upcoming_events`: Query events within specified timeframes

## Event Data Structure

Events are structured with the following fields:

```python
{
    "title": str,           # Event name/title
    "date": str,            # YYYY-MM-DD format
    "time": str,            # HH:MM format (24-hour)
    "location": str,        # Optional venue/location
    "attendees": List[str], # Optional list of attendee names
    "description": str,     # Optional additional details
    "created_at": str       # Timestamp of creation
}
```

## Natural Language Processing

The agent can interpret various time and date formats:

- **Relative dates**: "tomorrow", "next Monday", "in 3 days"
- **Time formats**: "2pm", "14:30", "9:00 AM"
- **Location extraction**: "at Downtown Office", "in Room 101"
- **Attendee parsing**: "with John and Sarah", "meeting with team"

## Configuration

Key configuration constants in the code:

```python
CALENDAR_FILE = "events.ics"      # ICS calendar output
CSV_FILE = "events_backup.csv"    # CSV backup file
EVENT_LOG = "event_log.txt"       # Activity log
```

## Error Handling

The system includes robust error handling:

- **Fallback parsing** when AI extraction fails
- **File creation** if storage files don't exist
- **Graceful degradation** for malformed inputs
- **Detailed error messages** for debugging

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages:
  - `langchain`
  - `langchain-openai`
  - `pydantic`
  - `ics`
  - `pathlib`

## File Structure

```
event-management-agent/
├── event_agent.py          # Main application file
├── events.ics             # Generated ICS calendar
├── events_backup.csv      # CSV backup storage
├── event_log.txt          # Activity log
└── README.md              # This file
```

## Examples

### Adding Events

```bash
# Simple event
"Lunch tomorrow at noon"

# Detailed event
"Schedule project review on July 30th at 3pm in Conference Room B with Alice, Bob, and Charlie"

# Recurring context
"Weekly team meeting every Thursday at 10am"
```

### Querying Events

```bash
# Time-based queries
"Show today's events"
"What's coming up this week?"
"List events for the next month"

# General queries
"Show my calendar"
"Upcoming events"
"What do I have scheduled?"
```

   ## Test Suite

The test suite (`test_event_agent.py`) verifies the core functionality of the Event Management AI Agent with the following test cases:

### Event Extraction Tests
1. **Basic Extraction**  
   - Verifies extraction of complete event details (title, date, time, location, attendees)
   - Example: "Schedule a meeting with Alice tomorrow at 3 PM in Conference Room A"

2. **Missing Time Handling**  
   - Tests default time assignment when not specified
   - Example: "Book a yoga class next Monday" → Defaults to 12:00

3. **Today/Tomorrow Handling**  
   - Validates relative date calculations
   - Example: "Set up a quick sync today" → Uses current date

4. **Event Validation**  
   - Ensures extracted data passes Pydantic model validation
   - Example: "Team meeting Friday at 2pm"

### Tool Functionality Tests
5. **Extract and Add Tool**  
   - Tests the end-to-end tool that extracts and stores events
   - Example: "Doctor appointment tomorrow at 3pm"

6. **Extract Only Tool**  
   - Verifies the preview functionality without storage
   - Example: "Team sync next Monday at 9 AM"

### Agent Integration Tests
7. **Event Creation Flow**  
   - Tests the full agent workflow for creating events
   - Mocks both extraction and storage components

8. **Event Listing Flow**  
   - Verifies the agent's ability to list upcoming events
   - Example: "List events for next week"

### Running Tests
```bash
python -m pytest test_event_agent.py -v
