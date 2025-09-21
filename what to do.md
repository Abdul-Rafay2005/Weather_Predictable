 What To Do – Will It Rain On My Parade?

This file explains who is working on what, so the project runs smoothly.

---

## 👩‍💻 ilsa – Frontend
- Setup React project (with Tailwind or Next.js/Vite).
- Create input form for **location + date**.
- Build UI components for:
  - Result display (rain probability, heat index, etc.)
  - Weather icons and simple summary.
- Add visualization (charts for trends, optional map).
- Connect to backend API endpoints and display data.
- Polish UI/UX (responsive, clean, easy to use).

---

## 👨‍💻 Abdul Rafay
– Backend
- Setup backend server (Node.js/Express or Python FastAPI).
- Build API endpoints:
  - `/forecast` → get near-term weather data.
  - `/climatology` → get historical data for selected location/date.
- Integrate external datasets:
  - NASA Earth science data.
  - Optionally NOAA/OpenWeatherMap for live forecasts.
- Process and clean data, return probabilities in JSON format.
- Write API documentation for frontend integration.

---

✅ Both sides will connect at the **API layer**:
- Frontend → sends location + date.
- Backend → responds with probabilities (rain %, heat %, etc.).
