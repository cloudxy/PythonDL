import api from './index'

export const gdpApi = {
  getList(params) {
    return api.get('/v1/consumption/gdp', { params })
  },
  getDetail(id) {
    return api.get(`/v1/consumption/gdp/${id}`)
  },
  create(data) {
    return api.post('/v1/consumption/gdp', data)
  },
  update(id, data) {
    return api.put(`/v1/consumption/gdp/${id}`, data)
  },
  delete(id) {
    return api.delete(`/v1/consumption/gdp/${id}`)
  },
  getTrend(params) {
    return api.get('/v1/consumption/gdp/trend', { params })
  }
}

export const populationApi = {
  getList(params) {
    return api.get('/v1/consumption/population', { params })
  },
  getDetail(id) {
    return api.get(`/v1/consumption/population/${id}`)
  },
  create(data) {
    return api.post('/v1/consumption/population', data)
  },
  update(id, data) {
    return api.put(`/v1/consumption/population/${id}`, data)
  },
  delete(id) {
    return api.delete(`/v1/consumption/population/${id}`)
  },
  getStatistics(params) {
    return api.get('/v1/consumption/population/statistics', { params })
  }
}

export const economicIndicatorApi = {
  getList(params) {
    return api.get('/v1/consumption/indicators', { params })
  },
  getDetail(id) {
    return api.get(`/v1/consumption/indicators/${id}`)
  },
  getComparison(params) {
    return api.get('/v1/consumption/indicators/comparison', { params })
  }
}

export const communityApi = {
  getList(params) {
    return api.get('/v1/consumption/community', { params })
  },
  getDetail(id) {
    return api.get(`/v1/consumption/community/${id}`)
  },
  getAnalysis(id) {
    return api.get(`/v1/consumption/community/${id}/analysis`)
  }
}

export const consumptionForecastApi = {
  getList(params) {
    return api.get('/v1/consumption/forecasts', { params })
  },
  getDetail(id) {
    return api.get(`/v1/consumption/forecasts/${id}`)
  },
  create(data) {
    return api.post('/v1/consumption/forecasts', data)
  },
  runForecast(id) {
    return api.post(`/v1/consumption/forecasts/${id}/run`)
  },
  getResults(id) {
    return api.get(`/v1/consumption/forecasts/${id}/results`)
  }
}
