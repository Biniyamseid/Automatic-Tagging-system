export async function predictTags(text: string): Promise<string[]> {
  try {
    const response = await fetch('https://automatic-tagging-system.onrender.com/predict_tags', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to predict tags');
    }
    
    const data = await response.json();
    return data.tags || [];
  } catch (error) {
    console.error('Error predicting tags:', error);
    return [];
  }
}