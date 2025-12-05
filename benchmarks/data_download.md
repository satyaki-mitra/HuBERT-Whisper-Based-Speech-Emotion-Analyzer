# FOR RAVDESS DATASET

# Create directory if needed
mkdir -p benchmarks/data

# Download RAVDESS
curl -L "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip" -o ravdess.zip

# Extract
unzip ravdess.zip -d benchmarks/data/ravdess/

# Clean up
rm ravdess.zip

