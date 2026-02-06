"""
MP3 to WAV Converter

This script converts MP3 files to WAV format using pydub.
Requires: pip install pydub
Also requires ffmpeg to be installed on your system.


Convert all MP3s in a directory:
bashpython mp3_to_wav_converter.py --dir /path/to/mp3s
# or specify output directory:
python mp3_to_wav_converter.py --dir /path/to/mp3s /path/to/output


Convert a single file:
bashpython mp3_to_wav_converter.py song.mp3
# or specify output name:
python mp3_to_wav_converter.py song.mp3 output.wav
"""

from pydub import AudioSegment
import os
import sys


def convert_mp3_to_wav(mp3_file, output_file=None):
    """
    Convert a single MP3 file to WAV format.
    
    Args:
        mp3_file (str): Path to the input MP3 file
        output_file (str, optional): Path to the output WAV file. 
                                    If None, uses same name with .wav extension
    
    Returns:
        str: Path to the output WAV file
    """
    # Check if input file exists
    if not os.path.exists(mp3_file):
        raise FileNotFoundError(f"Input file not found: {mp3_file}")
    
    # Check if input file is MP3
    if not mp3_file.lower().endswith('.mp3'):
        raise ValueError(f"Input file must be an MP3 file: {mp3_file}")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.splitext(mp3_file)[0] + '.wav'
    
    # Load the MP3 file
    print(f"Loading: {mp3_file}")
    audio = AudioSegment.from_mp3(mp3_file)
    
    # Export as WAV
    print(f"Converting to: {output_file}")
    audio.export(output_file, format='wav')
    
    print(f"✓ Conversion complete: {output_file}")
    return output_file


def convert_directory(directory, output_dir=None):
    """
    Convert all MP3 files in a directory to WAV format.
    
    Args:
        directory (str): Path to directory containing MP3 files
        output_dir (str, optional): Path to output directory. 
                                   If None, saves in same directory as source files
    """
    # Get all MP3 files in directory
    mp3_files = [f for f in os.listdir(directory) if f.lower().endswith('.mp3')]
    
    if not mp3_files:
        print(f"No MP3 files found in {directory}")
        return
    
    print(f"Found {len(mp3_files)} MP3 file(s)")
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert each file
    for mp3_file in mp3_files:
        input_path = os.path.join(directory, mp3_file)
        
        if output_dir:
            output_path = os.path.join(output_dir, os.path.splitext(mp3_file)[0] + '.wav')
        else:
            output_path = None
        
        try:
            convert_mp3_to_wav(input_path, output_path)
        except Exception as e:
            print(f"✗ Error converting {mp3_file}: {e}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python mp3_to_wav_converter.py input.mp3 [output.wav]")
        print("  Directory:   python mp3_to_wav_converter.py --dir /path/to/directory [/path/to/output]")
        sys.exit(1)
    
    if sys.argv[1] == "--dir":
        # Convert all files in directory
        if len(sys.argv) < 3:
            print("Error: Please specify a directory")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        convert_directory(input_dir, output_dir)
    
    else:
        # Convert single file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        try:
            convert_mp3_to_wav(input_file, output_file)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
