<template>
  <div class="flex -space-x-2">
    <div
      v-for="(avatar, index) in visibleAvatars"
      :key="index"
      :class="[
        'w-10 h-10 rounded-full border-2 border-white flex items-center justify-center text-sm font-semibold',
        index < maxVisible ? avatarColorClass(avatar.color) : 'bg-secondary-200 text-secondary-600'
      ]"
    >
      <img v-if="avatar.src" :src="avatar.src" :alt="avatar.name" class="w-full h-full rounded-full object-cover" />
      <span v-else>{{ avatar.initials || getInitials(avatar.name) }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  avatars: {
    type: Array,
    default: () => []
  },
  maxVisible: {
    type: Number,
    default: 4
  }
})

const visibleAvatars = computed(() => {
  const total = props.avatars.length
  if (total <= props.maxVisible) {
    return props.avatars
  }
  const visible = props.avatars.slice(0, props.maxVisible - 1)
  visible.push({ initials: `+${total - props.maxVisible + 1}`, color: 'secondary' })
  return visible
})

const getInitials = (name) => {
  if (!name) return 'U'
  return name.slice(0, 2).toUpperCase()
}

const avatarColorClass = (color = 'primary') => {
  const classes = {
    primary: 'bg-primary-100 text-primary-600',
    secondary: 'bg-secondary-100 text-secondary-600',
    success: 'bg-success-100 text-success-600',
    warning: 'bg-warning-100 text-warning-600',
    danger: 'bg-danger-100 text-danger-600'
  }
  return classes[color]
}
</script>
