import api from './index'

export const stockApi = {
  // 获取股票列表
  getList(params) {
    return api.get('/v1/finance/stocks', { params })
  },

  // 获取股票详情
  getDetail(id) {
    return api.get(`/v1/finance/stocks/${id}`)
  },

  // 创建股票
  create(data) {
    return api.post('/v1/finance/stocks', data)
  },

  // 更新股票
  update(id, data) {
    return api.put(`/v1/finance/stocks/${id}`, data)
  },

  // 删除股票
  delete(id) {
    return api.delete(`/v1/finance/stocks/${id}`)
  },

  // 获取股票历史数据
  getHistory(id, params) {
    return api.get(`/v1/finance/stocks/${id}/history`, { params })
  },

  // 同步股票数据
  syncData(id) {
    return api.post(`/v1/finance/stocks/${id}/sync`)
  }
}

export const stockPredictionApi = {
  // 获取预测列表
  getList(params) {
    return api.get('/v1/finance/predictions', { params })
  },

  // 获取预测详情
  getDetail(id) {
    return api.get(`/v1/finance/predictions/${id}`)
  },

  // 创建预测
  create(data) {
    return api.post('/v1/finance/predictions', data)
  },

  // 运行预测
  runPrediction(id) {
    return api.post(`/v1/finance/predictions/${id}/run`)
  },

  // 获取预测结果
  getResults(id) {
    return api.get(`/v1/finance/predictions/${id}/results`)
  }
}

export const riskAssessmentApi = {
  // 获取风险评估列表
  getList(params) {
    return api.get('/v1/finance/risk-assessments', { params })
  },

  // 获取风险评估详情
  getDetail(id) {
    return api.get(`/v1/finance/risk-assessments/${id}`)
  },

  // 创建风险评估
  create(data) {
    return api.post('/v1/finance/risk-assessments', data)
  },

  // 运行风险评估
  runAssessment(id) {
    return api.post(`/v1/finance/risk-assessments/${id}/run`)
  },

  // 获取评估报告
  getReport(id) {
    return api.get(`/v1/finance/risk-assessments/${id}/report`)
  }
}
