export interface Turn {
  speaker: "maya" | "marcus";
  line: string;
  start: number;    // seconds from section start
  duration: number; // seconds
}

export interface ConversationProps {
  title: string;
  audioFile: string; // filename in public/
  turns: Turn[];
  imageUrl?: string;  // optional topic image (filename in public/, or full URL)
  mayaColor?: string;
  marcusColor?: string;
}
