import os
import glob
from run_headless import run_headless

def main():
    tracks_dir = "racetrack-database/tracks"
    racelines_dir = "racetrack-database/racelines"
    
    # Get all csv files in tracks directory
    track_files = glob.glob(os.path.join(tracks_dir, "*.csv"))
    track_files.sort()
    
    results = []
    
    print(f"{'Track':<20} | {'Time':<10} | {'Violations':<10}")
    print("-" * 46)
    
    for track_path in track_files:
        filename = os.path.basename(track_path)
        track_name = os.path.splitext(filename)[0]
        
        raceline_path = os.path.join(racelines_dir, filename)
        
        if not os.path.exists(raceline_path):
            print(f"Skipping {track_name}: No raceline found at {raceline_path}")
            continue
            
        try:
            # print(f"Running {track_name}...")
            lap_time, violations = run_headless(track_path, raceline_path)
            
            if lap_time is not None:
                print(f"{track_name:<20} | {lap_time:<10.2f} | {violations:<10}")
                results.append((track_name, lap_time, violations))
            else:
                print(f"{track_name:<20} | {'DNF':<10} | {violations:<10}")
                results.append((track_name, float('inf'), violations))
                
        except Exception as e:
            print(f"{track_name:<20} | {'Error':<10} | {str(e)}")

if __name__ == "__main__":
    main()
