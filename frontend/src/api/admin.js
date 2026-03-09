import api from './index'

export const userApi = {
  // 获取用户列表
  getList(params) {
    return api.get('/v1/admin/users', { params })
  },

  // 获取用户详情
  getDetail(id) {
    return api.get(`/v1/admin/users/${id}`)
  },

  // 创建用户
  create(data) {
    return api.post('/v1/admin/users', data)
  },

  // 更新用户
  update(id, data) {
    return api.put(`/v1/admin/users/${id}`, data)
  },

  // 删除用户
  delete(id) {
    return api.delete(`/v1/admin/users/${id}`)
  },

  // 批量删除用户
  batchDelete(ids) {
    return api.post('/v1/admin/users/batch-delete', { ids })
  },

  // 修改用户状态
  updateStatus(id, status) {
    return api.patch(`/v1/admin/users/${id}/status`, { status })
  },

  // 重置用户密码
  resetPassword(id) {
    return api.post(`/v1/admin/users/${id}/reset-password`)
  }
}

export const roleApi = {
  // 获取角色列表
  getList(params) {
    return api.get('/v1/admin/roles', { params })
  },

  // 获取角色详情
  getDetail(id) {
    return api.get(`/v1/admin/roles/${id}`)
  },

  // 创建角色
  create(data) {
    return api.post('/v1/admin/roles', data)
  },

  // 更新角色
  update(id, data) {
    return api.put(`/v1/admin/roles/${id}`, data)
  },

  // 删除角色
  delete(id) {
    return api.delete(`/v1/admin/roles/${id}`)
  },

  // 获取角色权限
  getPermissions(id) {
    return api.get(`/v1/admin/roles/${id}/permissions`)
  },

  // 更新角色权限
  updatePermissions(id, permissionIds) {
    return api.put(`/v1/admin/roles/${id}/permissions`, { permission_ids: permissionIds })
  }
}

export const permissionApi = {
  // 获取权限列表
  getList(params) {
    return api.get('/v1/admin/permissions', { params })
  },

  // 获取权限树
  getTree() {
    return api.get('/v1/admin/permissions/tree')
  },

  // 创建权限
  create(data) {
    return api.post('/v1/admin/permissions', data)
  },

  // 更新权限
  update(id, data) {
    return api.put(`/v1/admin/permissions/${id}`, data)
  },

  // 删除权限
  delete(id) {
    return api.delete(`/v1/admin/permissions/${id}`)
  }
}

export const configApi = {
  // 获取配置列表
  getList(params) {
    return api.get('/v1/admin/configs', { params })
  },

  // 获取配置详情
  getDetail(id) {
    return api.get(`/v1/admin/configs/${id}`)
  },

  // 获取配置值
  getValue(key) {
    return api.get(`/v1/admin/configs/${key}/value`)
  },

  // 更新配置
  update(id, data) {
    return api.put(`/v1/admin/configs/${id}`, data)
  },

  // 批量更新配置
  batchUpdate(data) {
    return api.post('/v1/admin/configs/batch-update', data)
  }
}

export const logApi = {
  // 获取日志列表
  getList(params) {
    return api.get('/v1/admin/logs', { params })
  },

  // 获取日志详情
  getDetail(id) {
    return api.get(`/v1/admin/logs/${id}`)
  },

  // 导出日志
  export(params) {
    return api.get('/v1/admin/logs/export', { params, responseType: 'blob' })
  },

  // 清理日志
  clear(beforeDays) {
    return api.post('/v1/admin/logs/clear', { before_days: beforeDays })
  }
}

export const dashboardApi = {
  // 获取统计数据
  getStats() {
    return api.get('/v1/admin/dashboard/stats')
  },

  // 获取趋势数据
  getTrend(params) {
    return api.get('/v1/admin/dashboard/trend', { params })
  },

  // 获取最近活动
  getRecentActivities() {
    return api.get('/v1/admin/dashboard/activities')
  }
}
