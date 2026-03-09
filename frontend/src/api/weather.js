import api from './index'

export const weatherStationApi = {
  // 获取气象站点列表
  getList(params) {
    return api.get('/v1/weather/stations', { params })
  },

  // 获取气象站点详情
  getDetail(id) {
    return api.get(`/v1/weather/stations/${id}`)
  },

  // 创建气象站点
  create(data) {
    return api.post('/v1/weather/stations', data)
  },

  // 更新气象站点
  update(id, data) {
    return api.put(`/v1/weather/stations/${id}`, data)
  },

  // 删除气象站点
  delete(id) {
    return api.delete(`/v1/weather/stations/${id}`)
  }
}

export const weatherDataApi = {
  // 获取气象数据列表
  getList(params) {
    return api.get('/v1/weather/data', { params })
  },

  // 获取气象数据详情
  getDetail(id) {
    return api.get(`/v1/weather/data/${id}`)
  },

  // 导入气象数据
  import(data) {
    return api.post('/v1/weather/data/import', data)
  },

  // 导出气象数据
  export(params) {
    return api.get('/v1/weather/data/export', { params, responseType: 'blob' })
  }
}

export const weatherForecastApi = {
  // 获取气象预测列表
  getList(params) {
    return api.get('/v1/weather/forecasts', { params })
  },

  // 获取气象预测详情
  getDetail(id) {
    return api.get(`/v1/weather/forecasts/${id}`)
  },

  // 创建气象预测
  create(data) {
    return api.post('/v1/weather/forecasts', data)
  },

  // 运行预测
  runForecast(id) {
    return api.post(`/v1/weather/forecasts/${id}/run`)
  },

  // 获取预测结果
  getResults(id) {
    return api.get(`/v1/weather/forecasts/${id}/results`)
  }
}
