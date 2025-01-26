export interface Article {
  id: string;
  title: string;
  content: string;
  tags: string[];
  createdAt: string;
}

export interface TagResponse {
  tags: string[];
}