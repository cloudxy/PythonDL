import api from './index'

export const crawlerApi = {
  // 获取爬虫任务列表
  getList(params) {
    return api.get('/v1/crawler/tasks', { params })
  },

  // 获取爬虫任务详情
  getDetail(id) {
    return api.get(`/v1/crawler/tasks/${id}`)
  },

  // 创建爬虫任务
  create(data) {
    return api.post('/v1/crawler/tasks', data)
  },

  // 更新爬虫任务
  update(id, data) {
    return api.put(`/v1/crawler/tasks/${id}`, data)
  },

  // 删除爬虫任务
  delete(id) {
    return api.delete(`/v1/crawler/tasks/${id}`)
  },

  // 启动任务
  start(id) {
    return api.post(`/v1/crawler/tasks/${id}/start`)
  },

  // 停止任务
  stop(id) {
    return api.post(`/v1/crawler/tasks/${id}/stop`)
  },

  // 获取任务日志
  getLogs(id, params) {
    return api.get(`/v1/crawler/tasks/${id}/logs`, { params })
  },

  // 获取任务统计
  getStats(id) {
    return api.get(`/v1/crawler/tasks/${id}/stats`)
  }
}
