import { observable, action, decorate, runInAction } from "mobx"
import { TMDB_BEARER } from "../config"

const html = document.querySelector('html')

const headers = {
  "accept": "application/json",
  "Authorization": `Bearer ${TMDB_BEARER}`
}
 
// Store for fetching the Movies Page
class Movie {
  details = []
  credits = []
  loaded = false
  recommendations = []
  estimatedRevenue = null
  isEstimatedRevenue = false

  fetchAll(id) {
    runInAction(() => {
      this.loaded = false
    })

    // Fetch basic movie details
    fetch(
      `https://api.themoviedb.org/3/movie/${id}?language=en-US`,
      { headers }
    )
      .then(res => res.json())
      .then(res => {
        this.setDetails(res)
        if (this.details) {
          html.style.background = `url(https://image.tmdb.org/t/p/w1280${this.details.backdrop_path}) 
                center center / cover no-repeat fixed`
          
          // Always fetch revenue prediction for comparison
          console.log('Fetching revenue prediction for:', this.details.title);
          fetch('http://localhost:5000/predict_revenue', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            body: JSON.stringify({ title: this.details.title })
          })
          .then(res => {
            console.log('Revenue prediction response:', res);
            return res.json();
          })
          .then(data => {
            console.log('Revenue prediction data:', data);
            if (!data.error) {
              this.setEstimatedRevenue(data.revenue, data.isEstimate)
            }
          })
          .catch(err => console.error('Error predicting revenue:', err))
        }
        return this.details
    })

    fetch(
      `https://api.themoviedb.org/3/movie/${id}/credits`,
      { headers }
    )
      .then(res => res.json())
      .then(res => {
        this.setCredits(res)
      })

    fetch(
      `https://api.themoviedb.org/3/movie/${id}/recommendations?language=en-US&page=1`,
      { headers }
    )
    .then(res => res.json())
    .then(res => (
      this.setRecommendations(res)
    ))
  }

  setDetails(data) {
    this.details = data
  }

  setCredits(data) {
    this.credits = data
  }

  setRecommendations(data) {
    this.recommendations = data
    this.loaded = true
  }

  setEstimatedRevenue(revenue, isEstimate) {
    // Convert to number if it's a string
    this.estimatedRevenue = typeof revenue === 'string' ? parseFloat(revenue) : revenue;
    this.isEstimatedRevenue = isEstimate;
    console.log('Revenue set in store:', this.estimatedRevenue, 'isEstimate:', this.isEstimatedRevenue);
  }
}

decorate(Movie, {
  details: observable,
  credits: observable,
  loaded: observable,
  recommendations: observable,
  estimatedRevenue: observable,
  isEstimatedRevenue: observable,
  setDetails: action,
  setCredits: action,
  setRecommendations: action,
  setEstimatedRevenue: action
})

let movieStore = new Movie()

export default movieStore