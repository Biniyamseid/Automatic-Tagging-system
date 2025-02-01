export async function predictTags(text: string): Promise<string[]> {
  try {
    const response = await fetch('https://automatic-tagging-system.onrender.com/predict_tags', {
      // const response = await fetch('http://localhost:8000/predict_tags', { uncomment this to run locally
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      mode: 'cors',
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to predict tags: ${response.status}`);
    }
    
    const data = await response.json();
    return data.tags || [];
  } catch (error) {
    console.error('Error predicting tags:', error);
    throw error;
  }
}