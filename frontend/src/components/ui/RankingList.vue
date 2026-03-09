<template>
  <div class="card">
    <div class="p-4 border-b border-secondary-100">
      <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
    </div>
    <div class="divide-y divide-secondary-100">
      <div
        v-for="(item, index) in items"
        :key="index"
        class="flex items-center justify-between p-4 hover:bg-secondary-50 cursor-pointer transition-colors"
        @click="$emit('select', item)"
      >
        <div class="flex items-center gap-3">
          <div
            :class="[
              'w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium',
              item.avatarColor || 'bg-primary-100 text-primary-600'
            ]"
          >
            {{ item.avatar || item.title?.slice(0, 2) }}
          </div>
          <div>
            <p class="text-sm font-medium text-secondary-900">{{ item.title }}</p>
            <p v-if="item.subtitle" class="text-xs text-secondary-500">{{ item.subtitle }}</p>
          </div>
        </div>
        <div class="text-right">
          <p class="text-sm font-semibold" :class="item.valueColor || 'text-secondary-900'">
            {{ item.value }}
          </p>
          <p v-if="item.change" class="text-xs" :class="item.change >= 0 ? 'text-success-600' : 'text-danger-600'">
            {{ item.change >= 0 ? '+' : '' }}{{ item.change }}%
          </p>
        </div>
      </div>
    </div>
    <div v-if="showMore" class="p-3 text-center border-t border-secondary-100">
      <button class="text-sm text-primary-600 hover:text-primary-700 font-medium">
        查看更多
      </button>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  items: {
    type: Array,
    default: () => []
  },
  showMore: {
    type: Boolean,
    default: false
  }
})

defineEmits(['select'])
</script>
