# PPAnalyzer Frontend

React frontend with shadcn/ui components and Tailwind CSS.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Run development server:
```bash
npm run dev
```

Frontend runs at `http://localhost:3000`

## Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── ui/                 # shadcn/ui components
│   │   ├── button.jsx
│   │   ├── card.jsx
│   │   └── progress.jsx
│   ├── FileUpload.jsx      # File upload with drag & drop
│   ├── ProcessingStatus.jsx  # Analysis progress indicator
│   ├── Results.jsx         # Results display
│   └── VideoPlayer.jsx     # Video player component
├── lib/
│   └── utils.js            # Utility functions
├── App.jsx                 # Main app
├── main.jsx                # Entry point
└── index.css               # Global styles
```

## Components

### FileUpload
- Drag and drop interface
- File validation (PDF/HTML only)
- Visual feedback

### ProcessingStatus
- Step-by-step progress indicator
- Animated transitions
- Status messages

### Results
- Summary display
- Video player integration
- Interactive pie chart
- Category breakdown
- Sample sentence view

### VideoPlayer
- HTML5 video player
- HeyGen video integration

## Styling

- **Tailwind CSS** for utility-first styling
- **shadcn/ui** for component primitives
- Custom color scheme defined in `tailwind.config.js`

## API Integration

The frontend communicates with the backend via:
- Proxy configured in `vite.config.js`
- Axios for HTTP requests
- Base URL: `http://localhost:8000`

