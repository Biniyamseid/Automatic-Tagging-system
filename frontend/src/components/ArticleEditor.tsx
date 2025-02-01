import React, { useState } from 'react';
import { Pencil, Tags, RefreshCw } from 'lucide-react';
import { predictTags } from '../api';

export default function ArticleEditor() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateTags = async () => {
    if (!content) return;
    
    setIsLoading(true);
    setError(null);
    try {
      const predictedTags = await predictTags(content);
      setTags(predictedTags);
    } catch (error) {
      console.error('Failed to generate tags:', error);
      setError('Failed to generate tags. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePublish = () => {
    // TODO: Implement publishing functionality
    console.log({ title, content, tags });
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-8">
        <Pencil className="w-6 h-6 text-blue-600" />
        <h1 className="text-2xl font-bold text-gray-900">Create Article</h1>
      </div>

      <div className="space-y-4">
        <input
          type="text"
          placeholder="Article Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full px-4 py-2 text-xl font-semibold border-b-2 border-gray-200 focus:border-blue-600 outline-none"
        />

        <textarea
          placeholder="Write your article content here..."
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="w-full h-64 p-4 border rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none resize-none"
        />

        <div className="flex items-center space-x-4">
          <button
            onClick={handleGenerateTags}
            disabled={isLoading || !content}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Generate Tags</span>
          </button>

          {/* <button
            onClick={handlePublish}
            disabled={!title || !content}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            Publish
          </button> */}
        </div>

        {error && (
          <div className="text-red-600 text-sm mt-2">
            {error}
          </div>
        )}

        {tags.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center space-x-2 text-gray-600">
              <Tags className="w-4 h-4" />
              <span className="font-medium">Generated Tags:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {tags.map((tag, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}