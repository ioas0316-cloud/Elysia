import json
import os

def repair_memory(file_path):
    print(f"Reading {file_path} for repair...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the last valid 'ExperienceEvent' closing brace '}' in the 'stream' list
    # The stream list is at the end. We look for the last '},' or '}' that belongs to an event.
    
    # Let's find the position of the last valid event closing
    # We want to keep everything up to the last full event object in the list
    
    stream_start_idx = content.rfind('"stream": [')
    if stream_start_idx == -1:
        print("Could not find stream start. Memory might be severely corrupted.")
        return

    # Extract everything before the stream list
    head = content[:stream_start_idx + 11]
    
    # Extract the stream content and find the last valid event
    stream_body = content[stream_start_idx + 11:]
    
    # Find the last valid '},' which signifies an event has finished
    last_event_end = stream_body.rfind('    }')
    if last_event_end == -1:
        print("No valid events found to save in the stream.")
        # Just close the stream as empty
        fixed_content = head + "\n  ]\n}"
    else:
        # Keep up to the last valid '}'
        fixed_stream = stream_body[:last_event_end + 5]
        # Ensure it doesn't end with a trailing comma inside the list
        fixed_stream = fixed_stream.rstrip().rstrip(',')
        fixed_content = head + fixed_stream + "\n  ]\n}"

    # Verify if the new content is valid JSON
    try:
        data = json.loads(fixed_content)
        print(f"Success! Repaired JSON contains {len(data.get('stream', []))} events.")
        
        # Write back the fixed content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)
        print("Repair written to disk.")
    except Exception as e:
        print(f"Repair failed verification: {e}")
        # Emergency: try to recover at least the count and headers
        print("Attempting minimal recovery...")
        try:
            # Try to just close the file if it was close to finishing
            emergency_content = content.split('"stream": [')[0] + '"stream": []\n}'
            json.loads(emergency_content)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(emergency_content)
            print("Minimal recovery successful (memory stream lost but system restored).")
        except:
            print("Total structural failure. Manual intervention or backup restoration required.")

if __name__ == "__main__":
    path = "c:/Elysia/data/memory/experience/memory_state.json"
    repair_memory(path)
