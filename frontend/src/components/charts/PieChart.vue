<template>
  <div class="w-full h-full">
    <Pie v-if="loaded" :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Pie } from 'vue-chartjs'
import {
  Chart as ChartJS,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(
  ArcElement,
  Title,
  Tooltip,
  Legend
)

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  labels: {
    type: Array,
    required: true
  },
  data: {
    type: Array,
    required: true
  },
  colors: {
    type: Array,
    default: () => []
  },
  showLegend: {
    type: Boolean,
    default: true
  },
  showTooltip: {
    type: Boolean,
    default: true
  },
  donut: {
    type: Boolean,
    default: false
  },
  height: {
    type: Number,
    default: 300
  }
})

const loaded = ref(false)

onMounted(() => {
  loaded.value = true
})

const chartData = computed(() => ({
  labels: props.labels,
  datasets: [
    {
      data: props.data,
      backgroundColor: props.colors.length > 0 ? props.colors : getDefaultColors(),
      borderWidth: 2,
      borderColor: '#ffffff',
      hoverOffset: 10
    }
  ]
}))

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    title: {
      display: !!props.title,
      text: props.title,
      font: {
        size: 16,
        weight: 'bold'
      },
      padding: 20
    },
    legend: {
      display: props.showLegend,
      position: 'bottom',
      labels: {
        usePointStyle: true,
        pointStyle: 'rect',
        padding: 15
      }
    },
    tooltip: {
      enabled: props.showTooltip,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleFont: {
        size: 14
      },
      bodyFont: {
        size: 13
      },
      padding: 12,
      cornerRadius: 8,
      callbacks: {
        label: function(context) {
          const label = context.label || ''
          const value = context.parsed || 0
          const total = context.dataset.data.reduce((a, b) => a + b, 0)
          const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0
          return `${label}: ${value} (${percentage}%)`
        }
      }
    }
  }
}))

function getDefaultColors() {
  return [
    'rgba(59, 130, 246, 0.8)',
    'rgba(16, 185, 129, 0.8)',
    'rgba(245, 158, 11, 0.8)',
    'rgba(239, 68, 68, 0.8)',
    'rgba(139, 92, 246, 0.8)',
    'rgba(236, 72, 153, 0.8)',
    'rgba(14, 165, 233, 0.8)',
    'rgba(34, 197, 94, 0.8)',
    'rgba(251, 146, 60, 0.8)',
    'rgba(100, 116, 139, 0.8)'
  ]
}
</script>

<style scoped>
div {
  height: v-bind('height + "px"');
}
</style>
