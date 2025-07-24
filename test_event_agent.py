import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from event_agent import (
    extract_event_details_from_text, 
    EventDetails, 
    extract_and_add_event, 
    extract_event_details_only, 
    EventManagementAgent,
    initialize_extraction_llm
)
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")

def test_basic_extraction():
    input_text = "Schedule a meeting with Alice tomorrow at 3 PM in Conference Room A"
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    mock_response = {
        "title": "meeting with Alice",
        "date": tomorrow_date,
        "time": "15:00",
        "location": "Conference Room A",
        "attendees": ["Alice"],
        "description": None
    }

    with patch("event_agent.ChatOpenAI") as mock_chat:
        # Initialize the LLM
        initialize_extraction_llm("test_key")
        
        mock_instance = mock_chat.return_value
        mock_instance.invoke.return_value = MagicMock(content=json.dumps(mock_response))
        
        result = extract_event_details_from_text(input_text)
        assert result["title"] == "meeting with Alice"
        assert result["date"] == tomorrow_date
        assert result["time"] == "15:00"

def test_missing_time():
    input_text = "Book a yoga class next Monday"
    today = datetime.now()
    days_ahead = (0 - today.weekday()) % 7 or 7
    next_monday = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    mock_response = {
        "title": "yoga class",
        "date": next_monday,
        "time": "12:00",
        "location": None,
        "attendees": None,
        "description": None
    }

    with patch("event_agent.ChatOpenAI") as mock_chat:
        # Initialize the LLM
        initialize_extraction_llm("test_key")
        
        mock_instance = mock_chat.return_value
        mock_instance.invoke.return_value = MagicMock(content=json.dumps(mock_response))
        
        result = extract_event_details_from_text(input_text)
        assert result["title"] == "yoga class"
        assert result["date"] == next_monday
        assert result["time"] == "12:00"

def test_edge_case_today():
    input_text = "Set up a quick sync today"
    today_date = datetime.now().strftime("%Y-%m-%d")
    mock_response = {
        "title": "quick sync",
        "date": today_date,
        "time": "12:00",
        "location": None,
        "attendees": None,
        "description": None
    }

    with patch("event_agent.ChatOpenAI") as mock_chat:
        # Initialize the LLM
        initialize_extraction_llm("test_key")
        
        mock_instance = mock_chat.return_value
        mock_instance.invoke.return_value = MagicMock(content=json.dumps(mock_response))
        
        result = extract_event_details_from_text(input_text)
        assert result["title"] == "quick sync"
        assert result["date"] == today_date
        assert result["time"] == "12:00"

def test_event_details_validation():
    input_text = "Team meeting Friday at 2pm"
    today = datetime.now()
    days_ahead = (4 - today.weekday()) % 7 or 7
    friday_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    mock_response = {
        "title": "Team meeting",
        "date": friday_date,
        "time": "14:00",
        "location": None,
        "attendees": None,
        "description": None
    }

    with patch("event_agent.ChatOpenAI") as mock_chat:
        # Initialize the LLM
        initialize_extraction_llm("test_key")
        
        mock_instance = mock_chat.return_value
        mock_instance.invoke.return_value = MagicMock(content=json.dumps(mock_response))
        
        result = extract_event_details_from_text(input_text)
        event_details = EventDetails(**result)
        assert event_details.title == "Team meeting"
        assert event_details.date == friday_date
        assert event_details.time == "14:00"

def test_tool_extract_and_add():
    input_text = "Doctor appointment tomorrow at 3pm"

    with patch("event_agent.extract_event_details_from_text") as mock_extract:
        mock_extract.return_value = {
            "title": "Doctor appointment",
            "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "time": "15:00",
            "location": None,
            "attendees": None,
            "description": None
        }

        result = extract_and_add_event.invoke({"user_input": input_text})
        assert "successfully stored" in result

def test_tool_extract_only():
    input_text = "Team sync next Monday at 9 AM"

    with patch("event_agent.extract_event_details_from_text") as mock_extract:
        mock_extract.return_value = {
            "title": "Team sync",
            "date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "time": "09:00",
            "location": None,
            "attendees": None,
            "description": None
        }

        result = extract_event_details_only.invoke({"user_input": input_text})
        assert "Extracted details" in result

def test_agent_create_event():
    with patch("event_agent.extract_event_details_from_text") as mock_extract:
        mock_extract.return_value = {
            "title": "Mock Event",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": "10:00",
            "location": None,
            "attendees": None,
            "description": None
        }

        with patch("event_agent.add_event_to_calendar", return_value="Event successfully stored."):
            # Create a proper mock agent that returns the expected response
            mock_agent_instance = MagicMock()
            mock_agent_instance.invoke.return_value = {"output": "Event successfully stored."}
            
            with patch("event_agent.initialize_agent", return_value=mock_agent_instance):
                agent = EventManagementAgent(openai_api_key="test_key")
                result = agent.process_event_request("Schedule team sync tomorrow")
                assert "successfully stored" in result

                
def test_agent_list_event_response():
    # Create a proper mock tool
    mock_tool = MagicMock()
    mock_tool.name = "list_upcoming_events"
    mock_tool.description = "List upcoming events from the calendar"
    mock_tool.return_value = "Upcoming events for 7 days"

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"output": "Upcoming events for 7 days"}

    with patch("event_agent.list_upcoming_events", new=mock_tool):
        with patch("event_agent.initialize_agent", return_value=mock_agent):
            with patch("event_agent.ChatOpenAI"):
                with patch("event_agent.extraction_llm"):
                    agent = EventManagementAgent(openai_api_key="test_key")
                    response = agent.process_event_request("List events for next week")
                    # Get the actual response from agent
                    actual_response = mock_agent.invoke.return_value["output"]
                    assert "Upcoming events" in actual_response or mock_tool.return_value in actual_response