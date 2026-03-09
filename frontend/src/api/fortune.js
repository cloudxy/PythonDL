import api from './index'

export const fengShuiApi = {
  getList(params) {
    return api.get('/v1/fortune/fengshui', { params })
  },
  getDetail(id) {
    return api.get(`/v1/fortune/fengshui/${id}`)
  },
  create(data) {
    return api.post('/v1/fortune/fengshui', data)
  },
  update(id, data) {
    return api.put(`/v1/fortune/fengshui/${id}`, data)
  },
  delete(id) {
    return api.delete(`/v1/fortune/fengshui/${id}`)
  }
}

export const faceReadingApi = {
  getList(params) {
    return api.get('/v1/fortune/face-reading', { params })
  },
  getDetail(id) {
    return api.get(`/v1/fortune/face-reading/${id}`)
  },
  create(data) {
    return api.post('/v1/fortune/face-reading', data)
  },
  update(id, data) {
    return api.put(`/v1/fortune/face-reading/${id}`, data)
  },
  delete(id) {
    return api.delete(`/v1/fortune/face-reading/${id}`)
  },
  analyze(data) {
    return api.post('/v1/fortune/face-reading/analyze', data)
  }
}

export const baziApi = {
  getList(params) {
    return api.get('/v1/fortune/bazi', { params })
  },
  getDetail(id) {
    return api.get(`/v1/fortune/bazi/${id}`)
  },
  calculate(data) {
    return api.post('/v1/fortune/bazi/calculate', data)
  },
  analyze(id) {
    return api.post(`/v1/fortune/bazi/${id}/analyze`)
  }
}

export const zhouYiApi = {
  getList(params) {
    return api.get('/v1/fortune/zhouyi', { params })
  },
  getDetail(id) {
    return api.get(`/v1/fortune/zhouyi/${id}`)
  },
  divine(data) {
    return api.post('/v1/fortune/zhouyi/divine', data)
  }
}

export const constellationApi = {
  getList(params) {
    return api.get('/v1/fortune/constellation', { params })
  },
  getDetail(id) {
    return api.get(`/v1/fortune/constellation/${id}`)
  },
  getHoroscope(sign, params) {
    return api.get(`/v1/fortune/constellation/${sign}/horoscope`, { params })
  }
}

export const fortuneApi = {
  getList(params) {
    return api.get('/v1/fortune/luck', { params })
  },
  getDetail(id) {
    return api.get(`/v1/fortune/luck/${id}`)
  },
  analyze(data) {
    return api.post('/v1/fortune/luck/analyze', data)
  }
}
