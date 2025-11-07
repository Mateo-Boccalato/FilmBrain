import { observable, decorate, action, runInAction } from "mobx";
import { TMDB_BEARER } from "../config"

const headers = {
  "accept": "application/json",
  "Authorization": `Bearer ${TMDB_BEARER}`
}
// Store for fetching the Actor's Page
class Actor {
  actorDetails = []
  actorCredits = []
  loaded = false

  fetchAll(actorId) {
    runInAction(() => {
      this.loaded = false
    })

    fetch(
      `https://api.themoviedb.org/3/person/${actorId}?language=en-US`,
      { headers }
    )
      .then(res => res.json())
      .then(res => {
        this.setDetails(res)
      })

    fetch(
      `https://api.themoviedb.org/3/person/${actorId}/movie_credits?language=en-US`,
      { headers }
    )
      .then(res => res.json())
      .then(res => {
        this.setCredits(res)
      })
  }

  setDetails(data) {
    this.actorDetails = data
  }

  setCredits(data) {
    this.actorCredits = data
    this.loaded = true
  }
}

decorate(Actor, {
  actorDetails: observable,
  actorCredits: observable,
  loaded: observable,
  setCredits: action,
  setDetails: action
})

const actorStore = new Actor()

export default actorStore