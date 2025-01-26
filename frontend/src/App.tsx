import React from 'react';
import ArticleEditor from './components/ArticleEditor';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center">
              <span className="text-xl font-bold text-gray-900">ArticleAI</span>
            </div>
          </div>
        </div>
      </nav>
      <main className="py-10">
        <ArticleEditor />
      </main>
    </div>
  );
}

export default App;