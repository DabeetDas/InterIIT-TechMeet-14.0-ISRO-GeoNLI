# TRINETRA

Full-stack applicaton built with Next.js, Supabase, Tailwind, Maptiler, Zustand.. Features real-time chat for geographic analysis and comparison with beautiful UI.

Vercel Link: [Visit TRINETRA](https://trinetra-silk.vercel.app/)

https://github.com/team54techmeet14/m3-isro-geospatial

## Features

### AI Chat Functionality

- **Real-time Streaming Responses** Powered by a custom model architecture for fast interaction.
- **Instant Captioning** Start chatting immediately — captions and basic interactions work without authentication.
- **Visual Question Answering (VQA)** Authenticated users can ask questions about images and receive grounded, visual-aware responses.
- **Agriculture Query Section** A dedicated module for agriculture-related queries, complete with syntax-highlighted outputs for clarity.
- **GeoCompare Module** Compare geospatial images side-by-side with real-time typing indicators and responsive UI behavior.
- **Grounded Data Support** Responses can be grounded to user-provided data and images for higher accuracy and traceability.

### Modern UI/UX

- **Dark theme** with beautiful gradients
- **Real-time chat interface** with instant response to queries
- **Loading states** and error handling
- **Toast notifications** for user feedback

### Technical Features

- **Next.js 15** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **shadcn/ui** components
- **Server-side rendering** with Supabase SSR
- **Maptiler** for providing the user with a searchable map

## Quick Start

### Prerequisites

- Node.js 18+
- MapTiler api key
- Supabase anon key
- Supabase project link
- Supabase bucket key

### 1. Clone the Repository

```bash
git clone https://github.com/team54techmeet14/m3-isro-geospatial
cd TRINETRA
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Environment Setup

Create a `.env` file on the root with the following variables all the keys are given in the file :

```env
# Supabase
NEXT_PUBLIC_MAPTILER_KEY=your_supabase_url
NEXT_PUBLIC_SUPABASE_URL=your_supabase_anon_key
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anaon_key
NEXT_PUBLIC_SUPABASE_BUCKET=your_supabase_bucket
NEXT_PUBLIC_BACKEND_URL=backend_url

```

### 4. Start Development Server

```bash
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to see the application.

## Project Structure

```
chatgpt_clone/
├── app/                         # Next.js App Router
│   ├── (homepage)/              # Homepage with anonymous chat
│   ├── (homepage)/              # Homepage with anonymous chat
│   ├── agrichat/                # For Agrichat portal
│   ├── map/                     # For Selecting items from the map
│   └── tgCompare/               # For terrain geoanalysis chat
│   └── api/                     # For api request to the backend
├── components/                  # Reusable UI components
│   ├── ui/                      # shadcn/ui components
│   └── layout/                 # Chat sidebar components
├── lib/                         # Utility libraries
│   ├── supabase/                # Supabase client configuration
└── public/                      # Static assets
```
## Security Features

- **Secure session management**
- **Environment variable protection**

---
