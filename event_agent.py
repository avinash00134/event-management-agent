import os
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import re
import json
import csv
from ics import Calendar, Event as ICSEvent
from pathlib import Path
from typing import Dict

# Configuration
CALENDAR_FILE = "events.ics"
CSV_FILE = "events_backup.csv"
EVENT_LOG = "event_log.txt"

# Pydantic model for structured event data
class EventDetails(BaseModel):
    """Structured event details extracted from natural language."""
    title: str = Field(description="The title or name of the event")
    date: str = Field(description="Event date in YYYY-MM-DD format")
    time: str = Field(description="Event time in HH:MM format (24-hour)")
    location: Optional[str] = Field(default=None, description="Event location or venue")
    attendees: Optional[List[str]] = Field(default=None, description="List of attendee names")
    description: Optional[str] = Field(default=None, description="Additional event description")

class EventStorage:
    """Handles storage of events in both ICS calendar format and CSV backup."""
    
    def __init__(self):
        self.calendar_file = CALENDAR_FILE
        self.csv_file = CSV_FILE
        self.event_log = EVENT_LOG
        self.ensure_files_exist()
    

    def ensure_files_exist(self):
        """Ensure storage files exist with proper headers."""
        # Create CSV file with headers if not exists
        if not Path(self.csv_file).exists():
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "title", "date", "time", "location", "attendees", "description", "created_at"
                ])
                writer.writeheader()
        
        # Create empty ICS calendar if not exists
        if not Path(self.calendar_file).exists():
            with open(self.calendar_file, 'w') as f:
                f.write(Calendar().serialize())
    def list_events(self, from_date: str, to_date: str) -> list[dict]:
        events = []
        if not Path(self.csv_file).exists():
            return events

        with open(self.csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if from_date <= row["date"] <= to_date:
                    events.append(row)
        return events
    
    def add_event(self, event: EventDetails) -> str:
        """Add event to both calendar and CSV backup."""
        # Create ICS event
        ics_event = ICSEvent()
        ics_event.name = event.title
        ics_event.begin = f"{event.date} {event.time}:00"  # Add seconds for ICS format
        if event.location:
            ics_event.location = event.location
        if event.description:
            ics_event.description = event.description
        
        # Add to calendar
        calendar = Calendar()
        if Path(self.calendar_file).stat().st_size > 0:
            with open(self.calendar_file, 'r') as f:
                calendar = Calendar(f.read())
        
        calendar.events.add(ics_event)
        with open(self.calendar_file, 'w') as f:
            f.write(calendar.serialize())
        
        # Rest of the method remains the same...
        # Add to CSV backup
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "title", "date", "time", "location", "attendees", "description", "created_at"
            ])
            writer.writerow({
                "title": event.title,
                "date": event.date,
                "time": event.time,
                "location": event.location,
                "attendees": ", ".join(event.attendees) if event.attendees else None,
                "description": event.description,
                "created_at": datetime.now().isoformat()
            })
        
        # Log the event
        with open(self.event_log, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - Added event: {event.title}\n")
        
        return f"Event '{event.title}' successfully stored in calendar and backup."
# Initialize storage
event_storage = EventStorage()

def add_event_to_calendar(title: str, date: str, time: str, location: str = None, 
                         attendees: List[str] = None, description: str = None) -> str:
    """
    Add an event to the calendar system with storage in both ICS and CSV formats.
    """
    try:
        # Create event details object
        event_details = EventDetails(
            title=title,
            date=date,
            time=time,
            location=location,
            attendees=attendees,
            description=description
        )
        
        # Store the event
        return event_storage.add_event(event_details)
        
    except Exception as e:
        return f"Error adding event: {str(e)}"

# Global LLM instance for extraction
extraction_llm = None

def initialize_extraction_llm(api_key: str):
    """Initialize the global LLM for extraction."""
    global extraction_llm
    extraction_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )

def extract_event_details_from_text(user_input: str) -> dict:
    """Extract event details using the LLM."""
    if not extraction_llm:
        return {"error": "LLM not initialized"}
    
    extraction_prompt = f"""
    Extract event details from the following user input and return ONLY a valid JSON object:
    
    User Input: {user_input}
    Current Date: {datetime.now().strftime("%Y-%m-%d")}
    
    Instructions:
    - Extract title, date (YYYY-MM-DD), time (HH:MM in 24-hour format), location, attendees, and description
    - For relative dates like 'tomorrow', 'next Monday', calculate the actual date
    - For times like '3 PM', convert to 24-hour format (15:00)
    - If information is missing, use reasonable defaults or null
    - Return ONLY a JSON object with these exact fields: title, date, time, location, attendees, description
    - attendees should be a list of strings or null
    
    JSON:
    """
    
    try:
        response = extraction_llm.invoke(extraction_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            return data
        
        # Fallback parsing
        return fallback_parsing(user_input)
        
    except Exception as e:
        print(f"Error in extraction: {e}")
        return fallback_parsing(user_input)

def fallback_parsing(user_input: str) -> dict:
    """Fallback parsing when LLM extraction fails."""
    # Simple keyword-based extraction
    words = user_input.lower().split()
    
    # Extract title (try to find meaningful words)
    title_words = []
    skip_words = {'on', 'at', 'in', 'with', 'for', 'tomorrow', 'today', 'next', 'this'}
    for word in words[:5]: 
        if word not in skip_words and not re.match(r'\d', word):
            title_words.append(word)
    
    title = ' '.join(title_words) if title_words else 'New Event'
    
    date = datetime.now().strftime("%Y-%m-%d")
    time = "12:00"
    
    # Try to find time patterns
    time_patterns = [
        (r'(\d{1,2}):(\d{2})', lambda m: f"{int(m.group(1)):02d}:{m.group(2)}"),
        (r'(\d{1,2})\s*pm', lambda m: f"{int(m.group(1)) + 12 if int(m.group(1)) < 12 else int(m.group(1))}:00"),
        (r'(\d{1,2})\s*am', lambda m: f"{int(m.group(1)):02d}:00"),
    ]
    
    for pattern, formatter in time_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            try:
                time = formatter(match)
                break
            except:
                continue
    
    if 'tomorrow' in user_input.lower():
        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif 'today' in user_input.lower():
        date = datetime.now().strftime("%Y-%m-%d")
    
    return {
        "title": title,
        "date": date,
        "time": time,
        "location": None,
        "attendees": None,
        "description": None
    }


@tool()
def extract_and_add_event(user_input: str) -> str:
    """
    Extract event details from natural language text and add the event to calendar.
    This is a single-input tool that handles both extraction and adding the event.
    
    Args:
        user_input: Natural language description of an event
    
    Returns:
        Confirmation message with event details
    """
    try:
        extracted_data = extract_event_details_from_text(user_input)
        if "error" in extracted_data:
            return f"Error extracting event details: {extracted_data['error']}"
        
        event_details = EventDetails(**extracted_data)
        
        result = add_event_to_calendar(
            title=event_details.title,
            date=event_details.date,
            time=event_details.time,
            location=event_details.location,
            attendees=event_details.attendees,
            description=event_details.description
        )
        
        return result
        
    except Exception as e:
        return f"Error processing event: {str(e)}"

@tool()
def extract_event_details_only(user_input: str) -> str:
    """
    Extract structured event information from natural language text without adding to calendar.
    Use this tool to preview what would be extracted before adding an event.
    
    Args:
        user_input: Natural language description of an event
    
    Returns:
        String representation of extracted event details
    """
    try:
        extracted_data = extract_event_details_from_text(user_input)
        if "error" in extracted_data:
            return f"Error extracting event details: {extracted_data['error']}"
        
        event_details = EventDetails(**extracted_data)
        return f"Extracted details: {event_details.model_dump()}"
        
    except Exception as e:
        return f"Error extracting event details: {str(e)}"

@tool()
def list_upcoming_events(days: int = 7) -> str:
    """
    List upcoming events from the calendar within the specified number of days.
    
    Args:
        days: Number of days to look ahead (default: 7)
    
    Returns:
        Formatted string of upcoming events
    """
    try:
        today = datetime.now().date().isoformat()
        future_date = (datetime.now() + timedelta(days=days)).date().isoformat()
        
        events = event_storage.list_events(from_date=today, to_date=future_date)
        
        if not events:
            return f"No events found in the next {days} days."
        
        result = [f"Upcoming events (next {days} days):"]
        for i, event in enumerate(events, 1):
            event_time = datetime.strptime(event['time'], "%H:%M").strftime("%I:%M %p")
            result.append(
                f"{i}. {event['title']} on {event['date']} at {event_time}\n"
                f"   Location: {event['location'] or 'Not specified'}\n"
                f"   Description: {event['description'] or 'None'}"
            )
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error listing events: {str(e)}"

class EventManagementAgent:
    """Main agent class for event management."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the event management agent."""
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        initialize_extraction_llm(openai_api_key)
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        self.tools = [
            extract_and_add_event,
            extract_event_details_only,
            list_upcoming_events
        ]
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors="Check your output and make sure it conforms!",
            early_stopping_method="generate"
        )

    def process_event_request(self, user_input: str) -> str:
        """Process a user's event scheduling request."""
        try:
            if any(keyword in user_input.lower() for keyword in ["list", "show", "upcoming", "events"]):
                days = 7
                if "next week" in user_input.lower():
                    days = 7
                elif "next month" in user_input.lower():
                    days = 30
                elif "today" in user_input.lower():
                    days = 1
                
                return list_upcoming_events.invoke({"days": days})
            
            prompt = f"""The user wants to schedule an event. Use the extract_and_add_event tool with this exact input:
            
            {user_input}
            """
            
            response = self.agent.invoke({"input": prompt})
            return response.get("output", str(response))
            
        except Exception as e:
            return f"Error processing event request: {str(e)}"
        


def main():
    """Main function to run the event management agent."""
    print("Enhanced Event Management Agent Started!")
    print("=" * 50)
    print(f"Events will be stored in: {CALENDAR_FILE} and {CSV_FILE}")
    
    try:
        agent = EventManagementAgent()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Interactive mode
    print("\nInteractive mode - Enter event requests or queries (type 'quit' to exit):")
    print("Examples:")
    print("- Schedule a meeting with John tomorrow at 2pm")
    print("- Add dentist appointment on July 25 at 9:30 AM")
    print("- Show my upcoming events")
    print("- List events for next week")
    
    while True:
        user_input = input("\nEnter your request: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input:
            try:
                result = agent.process_event_request(user_input)
                print(f"\nResult: {result}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()